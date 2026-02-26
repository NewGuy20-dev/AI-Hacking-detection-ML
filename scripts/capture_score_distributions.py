#!/usr/bin/env python3
"""Capture real sub-model score distributions for meta stress-test generation."""
import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.stress_test.v14.models import ModelWrapper
from src.stress_test.v14.runner import BASE_WEIGHTS
from src.stress_test.v14.scenarios import (
    PayloadGenerator,
    URLGenerator,
    TimeSeriesGenerator,
    TabularGenerator,
)


MODEL_ORDER = ["payload", "url", "timeseries", "network", "host"]


def _build_generator(model_name: str, seed: int):
    if model_name == "payload":
        return PayloadGenerator(seed=seed)
    if model_name == "url":
        return URLGenerator(seed=seed)
    if model_name == "timeseries":
        return TimeSeriesGenerator(seed=seed)
    if model_name in {"network", "host"}:
        return TabularGenerator(seed=seed)
    raise ValueError(f"Unsupported model for score capture: {model_name}")


def _compute_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "std": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture model score distributions for meta generator.")
    parser.add_argument("--samples-per-model", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="configs/score_distributions.json")
    parser.add_argument(
        "--from-logs-dir",
        type=str,
        default=None,
        help="Optional stress-test run folder (contains *_YYYY-MM-DD.jsonl) to derive distributions.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    distributions: Dict[str, Dict[str, Dict[str, float]]] = {}

    if args.from_logs_dir:
        logs_dir = Path(args.from_logs_dir)
        if not logs_dir.exists():
            raise FileNotFoundError(f"Missing logs directory: {logs_dir}")
        for model_name in MODEL_ORDER:
            log_matches = sorted(logs_dir.glob(f"{model_name}_*.jsonl"))
            if not log_matches:
                raise FileNotFoundError(f"No log found for model '{model_name}' in {logs_dir}")
            # Prefer the non-failures log file.
            log_file = next((p for p in log_matches if not p.name.endswith("_failures.jsonl")), log_matches[0])
            attack_scores: List[float] = []
            benign_scores: List[float] = []
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    score = float(row["confidence"])
                    if int(row["expected"]) == 1:
                        attack_scores.append(score)
                    else:
                        benign_scores.append(score)
            distributions[model_name] = {
                "attack": _compute_stats(attack_scores),
                "benign": _compute_stats(benign_scores),
            }
            print(
                f"[+] {model_name} from logs: benign mean={distributions[model_name]['benign']['mean']:.4f}, "
                f"attack mean={distributions[model_name]['attack']['mean']:.4f}"
            )
    else:
        for model_name in MODEL_ORDER:
            print(f"\n[+] Capturing scores for {model_name}...")
            wrapper = ModelWrapper(model_name).load()
            generator = _build_generator(model_name, args.seed)
            weights = BASE_WEIGHTS.get(model_name, {})

            if model_name in {"network", "host"}:
                scenarios = generator.generate(model_name, args.samples_per_model, weights, benign_ratio=0.5)
            else:
                scenarios = generator.generate(args.samples_per_model, weights, benign_ratio=0.5)

            attack_scores: List[float] = []
            benign_scores: List[float] = []

            for scenario in scenarios:
                _pred, conf, _lat = wrapper.predict(scenario.input_data)
                score = float(conf)
                if scenario.expected_label == 1:
                    attack_scores.append(score)
                else:
                    benign_scores.append(score)

            distributions[model_name] = {
                "attack": _compute_stats(attack_scores),
                "benign": _compute_stats(benign_scores),
            }
            print(
                f"    benign mean={distributions[model_name]['benign']['mean']:.4f}, "
                f"attack mean={distributions[model_name]['attack']['mean']:.4f}"
            )

    output_path.write_text(json.dumps(distributions, indent=2), encoding="utf-8")
    print(f"\nSaved score distributions to {output_path}")


if __name__ == "__main__":
    main()
