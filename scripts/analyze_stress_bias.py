#!/usr/bin/env python3
"""Analyze stress-test JSONL logs for confusion/FP/FN rates."""
import argparse
from pathlib import Path

from jsonl_utils import iter_records


def analyze(path: Path):
    tp = tn = fp = fn = 0
    benign_total = benign_fp = 0
    mal_total = mal_fn = 0

    for rec in iter_records(path):
        exp = rec.get('expected')
        pred = rec.get('predicted')
        if exp is None or pred is None or pred == -1:
            continue
        if exp == 1 and pred == 1:
            tp += 1
        elif exp == 0 and pred == 0:
            tn += 1
        elif exp == 0 and pred == 1:
            fp += 1
        elif exp == 1 and pred == 0:
            fn += 1

        if exp == 0:
            benign_total += 1
            if pred == 1:
                benign_fp += 1
        if exp == 1:
            mal_total += 1
            if pred == 0:
                mal_fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    benign_fp_rate = (benign_fp / benign_total) if benign_total else 0.0
    mal_fn_rate = (mal_fn / mal_total) if mal_total else 0.0

    return {
        'total': total,
        'accuracy': acc,
        'benign_total': benign_total,
        'benign_fp': benign_fp,
        'benign_fp_rate': benign_fp_rate,
        'mal_total': mal_total,
        'mal_fn': mal_fn,
        'mal_fn_rate': mal_fn_rate,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default='evaluation/stress_test_v14', help='Directory with JSONL logs')
    parser.add_argument('--date', default=None, help='Run date (YYYY-MM-DD) to filter')
    args = parser.parse_args()

    base = Path(args.dir)
    if not base.exists():
        raise SystemExit(f"Missing directory: {base}")

    paths = [p for p in sorted(base.glob('*.jsonl')) if not p.name.endswith('_failures.jsonl')]
    if args.date:
        paths = [p for p in paths if args.date in p.name]

    for path in paths:
        stats = analyze(path)
        model = path.stem.split('_')[0]
        print(f"\n{model.upper()} {path.name}")
        print(f"  total={stats['total']} acc={stats['accuracy']*100:.2f}%")
        print(f"  benign_total={stats['benign_total']} benign_fp={stats['benign_fp']} benign_FP_rate={stats['benign_fp_rate']*100:.2f}%")
        print(f"  mal_total={stats['mal_total']} mal_fn={stats['mal_fn']} mal_FN_rate={stats['mal_fn_rate']*100:.2f}%")
        print(f"  tp={stats['tp']} tn={stats['tn']} fp={stats['fp']} fn={stats['fn']}")


if __name__ == '__main__':
    main()
