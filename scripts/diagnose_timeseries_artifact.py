#!/usr/bin/env python3
"""Diagnose timeseries artifact separability using saved model + normalization."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stress_test.v14.models import ModelWrapper
from src.stress_test.v14.scenarios import TimeSeriesGenerator


def _load_fixture(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"Unexpected fixture shape for {path}: {arr.shape}")
    return arr


def _parse_category_weights(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Category weights must be a JSON object.")
    return {str(k): float(v) for k, v in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose saved timeseries artifact behavior.")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument(
        "--ddos-fixture",
        type=str,
        default="configs/stress_test/scenarios_v14/fixtures/timeseries_ddos.json",
    )
    parser.add_argument(
        "--normal-fixture",
        type=str,
        default="configs/stress_test/scenarios_v14/fixtures/timeseries_normal.json",
    )
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--stress-sample-count", type=int, default=0)
    parser.add_argument("--stress-benign-ratio", type=float, default=0.7)
    parser.add_argument("--stress-weights", type=str, default=None)
    args = parser.parse_args()

    wrapper = ModelWrapper("timeseries", Path(args.models_dir)).load()
    ddos = _load_fixture(Path(args.ddos_fixture))
    normal = _load_fixture(Path(args.normal_fixture))

    prepared_ddos = wrapper.preprocess(ddos).detach().cpu().numpy()
    prepared_normal = wrapper.preprocess(normal).detach().cpu().numpy()

    ddos_pred, ddos_conf, _ = wrapper.predict(ddos)
    normal_pred, normal_conf, _ = wrapper.predict(normal)

    report = {
        "model_artifact": wrapper.artifact_metadata,
        "fixtures": {
            "ddos": {
                "raw_shape": list(ddos.shape),
                "prepared_shape": list(prepared_ddos.shape),
                "prepared_min": float(prepared_ddos.min()),
                "prepared_max": float(prepared_ddos.max()),
                "prediction": int(ddos_pred),
                "confidence": float(ddos_conf),
            },
            "normal": {
                "raw_shape": list(normal.shape),
                "prepared_shape": list(prepared_normal.shape),
                "prepared_min": float(prepared_normal.min()),
                "prepared_max": float(prepared_normal.max()),
                "prediction": int(normal_pred),
                "confidence": float(normal_conf),
            },
        },
        "confidence_gap": float(abs(ddos_conf - normal_conf)),
    }

    if args.stress_sample_count > 0:
        weights = _parse_category_weights(args.stress_weights)
        generator = TimeSeriesGenerator(seed=42)
        scenarios = generator.generate(
            args.stress_sample_count,
            category_weights=weights,
            benign_ratio=args.stress_benign_ratio,
        )
        stress_summary: dict[str, dict[str, float]] = {}
        for scenario in scenarios:
            pred, conf, _ = wrapper.predict(scenario.input_data)
            key = f"{scenario.category}:{scenario.expected_label}"
            bucket = stress_summary.setdefault(key, {"count": 0, "conf_sum": 0.0, "predicted_attack": 0})
            bucket["count"] += 1
            bucket["conf_sum"] += float(conf)
            bucket["predicted_attack"] += int(pred == 1)
        report["stress_probe"] = {
            "sample_count": int(args.stress_sample_count),
            "benign_ratio": float(args.stress_benign_ratio),
            "weights": weights,
            "categories": {
                key: {
                    "count": int(stats["count"]),
                    "mean_confidence": float(stats["conf_sum"] / max(stats["count"], 1)),
                    "predicted_attack_rate": float(stats["predicted_attack"] / max(stats["count"], 1)),
                }
                for key, stats in sorted(stress_summary.items())
            },
        }
    print(json.dumps(report, indent=2))

    if (
        abs(ddos_conf - normal_conf) < args.epsilon
        or max(ddos_conf, normal_conf) < args.epsilon
        or min(ddos_conf, normal_conf) > 1.0 - args.epsilon
    ):
        raise SystemExit("Collapsed timeseries artifact: benign and attack probes are not separable.")


if __name__ == "__main__":
    main()
