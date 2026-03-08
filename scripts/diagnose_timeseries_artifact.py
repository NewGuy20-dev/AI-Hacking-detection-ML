#!/usr/bin/env python3
"""Diagnose timeseries artifact separability using saved model + normalization."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stress_test.v14.models import ModelWrapper


def _load_fixture(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"Unexpected fixture shape for {path}: {arr.shape}")
    return arr


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
    print(json.dumps(report, indent=2))

    if (
        abs(ddos_conf - normal_conf) < args.epsilon
        or max(ddos_conf, normal_conf) < args.epsilon
        or min(ddos_conf, normal_conf) > 1.0 - args.epsilon
    ):
        raise SystemExit("Collapsed timeseries artifact: benign and attack probes are not separable.")


if __name__ == "__main__":
    main()
