#!/usr/bin/env python3
"""Compare shadow logs against a candidate model to measure drift/disagreement."""
import argparse
import json
from pathlib import Path
from datetime import date

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stress_test.v14.models import ModelWrapper
from jsonl_utils import iter_records


def parse_args():
    p = argparse.ArgumentParser(description="Shadow log comparison")
    p.add_argument('--model', required=True, help='Model name to evaluate')
    p.add_argument('--log', required=True, help='Shadow log JSONL file')
    p.add_argument('--output', default=None, help='Output summary json')
    return p.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log)
    out_path = Path(args.output) if args.output else log_path.with_name(f"shadow_compare_{date.today().isoformat()}.json")

    wrapper = ModelWrapper(args.model, models_dir=Path('models')).load()

    total = 0
    disagreements = 0
    latencies = []
    confidences = []

    for rec in iter_records(log_path):
        if rec.get('input_type') != 'text':
            continue
        raw_input = rec.get('raw_input')
        if raw_input is None:
            continue
        pred, conf, lat = wrapper.predict(raw_input)
        orig_pred = rec.get('prediction')
        total += 1
        if pred != orig_pred:
            disagreements += 1
        latencies.append(lat)
        confidences.append(conf)

    summary = {
        'model': args.model,
        'log': str(log_path),
        'total': total,
        'disagreements': disagreements,
        'disagreement_rate': disagreements / total if total else 0.0,
        'latency_mean_ms': sum(latencies)/len(latencies) if latencies else 0.0,
        'run_date': date.today().isoformat(),
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Shadow compare complete: {summary}")


if __name__ == "__main__":
    main()
