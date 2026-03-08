"""Shared training helpers for remediation-oriented model training."""
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


def stratified_index_split(labels, test_size: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    indices = np.arange(labels.shape[0])
    train_idx, val_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return train_idx, val_idx


def binary_metrics(probabilities, labels, threshold: float) -> Dict[str, float]:
    probs = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    truth = np.asarray(labels, dtype=np.int32).reshape(-1)
    preds = (probs > float(threshold)).astype(np.int32)
    tp = int(np.sum((preds == 1) & (truth == 1)))
    tn = int(np.sum((preds == 0) & (truth == 0)))
    fp = int(np.sum((preds == 1) & (truth == 0)))
    fn = int(np.sum((preds == 0) & (truth == 1)))
    total = max(int(truth.size), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    accuracy = (tp + tn) / total
    f1 = (2 * precision * recall) / max(precision + recall, 1e-8)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


def load_operational_threshold(model_name: str, default: float = 0.5) -> float:
    project_root = Path(__file__).resolve().parents[2]
    threshold = float(default)

    yaml_path = project_root / "config" / "model_thresholds.yaml"
    if yaml_path.exists():
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict) and model_name in data:
            threshold = float(data[model_name])

    json_path = project_root / "configs" / "inference" / "optimal_thresholds.json"
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        mapping = {
            "payload": "payload_cnn",
            "url": "url_cnn",
            "timeseries": "timeseries",
            "network": "network",
            "fraud": "fraud",
            "meta": "meta",
            "host": "host",
            "anomaly": "anomaly",
        }
        key = mapping.get(model_name, model_name)
        if isinstance(payload, dict):
            thresholds = payload.get("thresholds", {})
            if isinstance(thresholds, dict) and key in thresholds:
                threshold = float(thresholds[key])
    return threshold


def write_training_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
