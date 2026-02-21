#!/usr/bin/env python3
"""Calibrate per-model thresholds using stress-test JSONL logs."""
import argparse
import json
from pathlib import Path
import numpy as np
import yaml


def iter_records(path):
    with Path(path).open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def best_threshold(records, fp_cap, recall_min):
    probs = []
    labels = []
    for rec in records:
        exp = rec.get('expected')
        conf = rec.get('confidence')
        if exp is None or conf is None:
            continue
        labels.append(int(exp))
        probs.append(float(conf))

    if not probs:
        return 0.5, 0.0, 0.0

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    thresholds = np.linspace(0.01, 0.99, 99)
    best = (0.5, -1.0, 0.0)

    for th in thresholds:
        preds = (probs > th).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        benign_total = tn + fp
        mal_total = tp + fn
        fp_rate = (fp / benign_total) if benign_total else 0.0
        recall = (tp / mal_total) if mal_total else 0.0

        if fp_rate > fp_cap or recall < recall_min:
            continue

        precision = (tp / (tp + fp)) if (tp + fp) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        if f1 > best[1]:
            best = (th, f1, recall)

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='evaluation/stress_test_v14')
    parser.add_argument('--date', default=None)
    parser.add_argument('--fp-cap', type=float, default=0.35)
    parser.add_argument('--recall-min', type=float, default=0.8)
    parser.add_argument('--output', default='config/model_thresholds.yaml')
    args = parser.parse_args()

    base = Path(args.dir)
    paths = sorted(base.glob('*.jsonl'))
    if args.date:
        paths = [p for p in paths if args.date in p.name and not p.name.endswith('_failures.jsonl')]

    thresholds = {}
    for path in paths:
        if path.name.endswith('_failures.jsonl'):
            continue
        model = path.stem.split('_')[0]
        records = list(iter_records(path))
        th, f1, recall = best_threshold(records, args.fp_cap, args.recall_min)
        thresholds[model] = round(float(th), 4)
        print(f"{model}: threshold={th:.3f} f1={f1:.3f} recall={recall:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(thresholds, f, sort_keys=True)


if __name__ == '__main__':
    main()
