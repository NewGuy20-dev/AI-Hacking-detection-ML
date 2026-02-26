#!/usr/bin/env python3
"""Diagnose timeseries model collapse using stress-test confidences."""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def _diagnose(attack: Dict[str, float], benign: Dict[str, float]) -> str:
    if attack["max"] < 0.1 and benign["max"] < 0.1:
        return "H1: sigmoid saturation likely (all outputs near zero)."
    if attack["mean"] < 0.3 and benign["mean"] > 0.6:
        return "H2: label inversion likely (attack/benign encoding mismatch)."
    if abs(attack["mean"] - 0.5) < 0.1 and abs(benign["mean"] - 0.5) < 0.1:
        return "H3: feature pipeline mismatch likely (uninformative ~0.5 outputs)."
    return "Inconclusive: requires feature-pipeline parity check + retraining diagnostics."


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose timeseries sigmoid output behavior.")
    parser.add_argument(
        "--input-log",
        type=str,
        default="evaluation/stress_test_v14/2026-02-25/timeseries_2026-02-25.jsonl",
        help="Path to timeseries stress-test JSONL log.",
    )
    args = parser.parse_args()

    log_path = Path(args.input_log)
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log file: {log_path}")

    attack_scores: List[float] = []
    benign_scores: List[float] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            score = float(row["confidence"])
            if int(row["expected"]) == 1:
                attack_scores.append(score)
            else:
                benign_scores.append(score)

    attack = _summarize(attack_scores)
    benign = _summarize(benign_scores)
    diagnosis = _diagnose(attack, benign)

    bins = [0.0, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
    attack_hist, _ = np.histogram(np.array(attack_scores, dtype=np.float32), bins=bins)
    benign_hist, _ = np.histogram(np.array(benign_scores, dtype=np.float32), bins=bins)

    print(f"Input log: {log_path}")
    print("ATTACK outputs :", attack)
    print("BENIGN outputs :", benign)
    print("ATTACK hist    :", attack_hist.tolist())
    print("BENIGN hist    :", benign_hist.tolist())
    print("Diagnosis      :", diagnosis)


if __name__ == "__main__":
    main()
