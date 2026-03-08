#!/usr/bin/env python3
"""Train network intrusion model with adversarial-benign augmentation."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.data_guardrails import assert_allowed_training_paths
from src.training.training_utils import binary_metrics, load_operational_threshold, write_training_manifest

FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
]


def _load_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in tqdm(handle, desc=f"Loading {path.name}"):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _augment_adversarial_benign(normal_df: pd.DataFrame, ratio: float = 0.3) -> pd.DataFrame:
    if normal_df.empty:
        return normal_df.copy()
    sample = normal_df.sample(frac=min(max(ratio, 0.0), 1.0), random_state=42, replace=False).copy()
    noisy_features = [col for col in FEATURES if col in sample.columns]
    for col in noisy_features:
        values = sample[col].astype(float).values
        if col in {"src_bytes", "dst_bytes", "count", "srv_count"}:
            sample[col] = np.clip(values * np.random.uniform(1.1, 1.6, size=len(sample)), a_min=0.0, a_max=None)
        else:
            sample[col] = np.clip(values + np.random.normal(0.0, 0.15, size=len(sample)), a_min=0.0, a_max=None)
    sample["label"] = 0
    sample["augmentation"] = "adversarial_benign"
    return sample


def main():
    base = Path(__file__).parent.parent
    synth_path = base / "datasets/network_intrusion/synthetic_500k.jsonl"
    live_benign_path = base / "datasets/live_benign/mawi_network_kdd.jsonl"
    assert_allowed_training_paths(
        [synth_path, live_benign_path],
        context="network training data source",
    )

    samples = _load_jsonl(synth_path) + _load_jsonl(live_benign_path)
    if not samples:
        raise FileNotFoundError("No network intrusion training data found.")

    df = pd.DataFrame(samples)
    if "label" not in df.columns:
        raise ValueError("Network training data must contain a 'label' column.")
    df["label"] = df["label"].fillna(0).astype(int)
    if "category" not in df.columns:
        df["category"] = np.where(df["label"] == 1, "attack", "normal")

    normal_df = df[df["label"] == 0].copy()
    augmented_benign = _augment_adversarial_benign(normal_df, ratio=0.3)
    train_df = pd.concat([df, augmented_benign], ignore_index=True, sort=False)

    X = train_df[[f for f in FEATURES if f in train_df.columns]].fillna(0).values.astype(np.float32)
    y = train_df["label"].values.astype(int)
    category = train_df["category"].astype(str).values
    stratify_key = np.array([f"{label}:{cat}" for label, cat in zip(y, category)], dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_key,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training RandomForest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=24,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    threshold = load_operational_threshold("network", default=0.70)
    metrics = binary_metrics(probs, y_test, threshold)
    print(
        f"Operational metrics@{threshold:.2f}: "
        f"f1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}, fpr={metrics['fpr']:.4f}"
    )

    models_dir = base / "models"
    joblib.dump(model, models_dir / "network_intrusion_model.pkl")
    joblib.dump(scaler, models_dir / "network_scaler.pkl")
    write_training_manifest(
        models_dir / "network_intrusion_training_manifest.json",
        {
            "model": "network",
            "dataset_size": int(len(train_df)),
            "operational_threshold": threshold,
            "metrics": metrics,
            "class_counts": train_df["label"].value_counts().to_dict(),
            "category_counts": train_df["category"].value_counts().to_dict(),
            "adversarial_benign_rows": int(len(augmented_benign)),
        },
    )
    print("✓ Saved network_intrusion_model.pkl")


if __name__ == "__main__":
    main()
