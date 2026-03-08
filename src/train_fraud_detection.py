#!/usr/bin/env python3
"""Train fraud detection with category-aware weighting and reproducible manifests."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.data_guardrails import assert_allowed_training_paths
from src.training.training_utils import binary_metrics, load_operational_threshold, write_training_manifest

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    XGBClassifier = None


def build_model(scale_pos_weight: float):
    """Create the preferred fraud classifier for the current environment."""
    if XGBClassifier is not None:
        print("Training XGBoost...")
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=max(scale_pos_weight, 1.0),
            subsample=0.9,
            colsample_bytree=0.9,
        )

    print("XGBoost not available; falling back to HistGradientBoostingClassifier.")
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=42,
    )


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


def _load_training_frame(base: Path) -> pd.DataFrame:
    csv_path = base / "datasets/fraud_detection/creditcard.csv"
    synth_path = base / "datasets/fraud_detection/synthetic_500k.jsonl"
    live_benign_path = base / "datasets/live_benign/fraud_benign.jsonl"
    augmented_path = base / "datasets/fraud_detection/augmented_fraud_categories.jsonl"
    assert_allowed_training_paths(
        [csv_path, synth_path, live_benign_path, augmented_path],
        context="fraud training data source",
    )

    frames = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "fraud_category" not in df.columns:
            df["fraud_category"] = np.where(df["Class"].astype(int) == 1, "generic_fraud", "normal")
        frames.append(df)

    jsonl_rows = _load_jsonl(synth_path) + _load_jsonl(live_benign_path) + _load_jsonl(augmented_path)
    if jsonl_rows:
        frames.append(pd.DataFrame(jsonl_rows))

    if not frames:
        raise FileNotFoundError("No fraud training data found.")

    df = pd.concat(frames, ignore_index=True, sort=False)
    if "Class" not in df.columns:
        raise ValueError("Fraud training data must contain a 'Class' column.")

    df["Class"] = df["Class"].fillna(0).astype(int)
    if "fraud_category" not in df.columns:
        df["fraud_category"] = np.where(df["Class"] == 1, "generic_fraud", "normal")
    df["fraud_category"] = df["fraud_category"].fillna(np.where(df["Class"] == 1, "generic_fraud", "normal"))
    return df


def _sample_weights(df: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(df), dtype=np.float32)
    fraud_mask = df["Class"].astype(int).values == 1
    neg = max(int((~fraud_mask).sum()), 1)
    pos = max(int(fraud_mask.sum()), 1)
    weights[fraud_mask] *= float(neg / pos)

    category_weight = {
        "card_not_present": 3.0,
        "account_takeover": 2.0,
        "synthetic": 1.2,
    }
    categories = df["fraud_category"].astype(str).values
    for category, boost in category_weight.items():
        weights[categories == category] *= boost
    return weights


def main():
    base = Path(__file__).parent.parent
    df = _load_training_frame(base)
    print(f"Samples: {len(df):,}, Fraud: {int(df['Class'].sum()):,}, Normal: {int((df['Class']==0).sum()):,}")

    features = [c for c in df.columns if c not in ["Class", "fraud_category"]]
    X = df[features].fillna(0).values.astype(np.float32)
    y = df["Class"].values.astype(int)
    categories = df["fraud_category"].astype(str).values
    stratify_key = np.array([f"{label}:{category}" for label, category in zip(y, categories)], dtype=object)
    weights = _sample_weights(df)

    X_train, X_test, y_train, y_test, w_train, _w_test = train_test_split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=42,
        stratify=stratify_key,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    neg_count = max(int((y_train == 0).sum()), 1)
    pos_count = max(int((y_train == 1).sum()), 1)
    model = build_model(scale_pos_weight=neg_count / pos_count)
    model.fit(X_train, y_train, sample_weight=w_train)

    probs = model.predict_proba(X_test)[:, 1]
    threshold = load_operational_threshold("fraud", default=0.75)
    metrics = binary_metrics(probs, y_test, threshold)
    print(f"Accuracy@0.5: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(
        f"Operational metrics@{threshold:.2f}: "
        f"f1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}, fpr={metrics['fpr']:.4f}"
    )

    models_dir = base / "models"
    joblib.dump(model, models_dir / "fraud_detection_model.pkl")
    joblib.dump(scaler, models_dir / "fraud_scaler.pkl")
    write_training_manifest(
        models_dir / "fraud_detection_training_manifest.json",
        {
            "model": "fraud",
            "dataset_size": int(len(df)),
            "features": features,
            "operational_threshold": threshold,
            "metrics": metrics,
            "category_counts": df["fraud_category"].value_counts().to_dict(),
            "class_counts": df["Class"].value_counts().to_dict(),
        },
    )
    print("✓ Saved fraud_detection_model.pkl")


if __name__ == "__main__":
    main()
