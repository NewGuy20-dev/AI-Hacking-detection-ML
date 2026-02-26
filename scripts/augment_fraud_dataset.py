#!/usr/bin/env python3
"""Augment fraud training data for card-not-present and account-takeover."""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_guardrails import assert_allowed_training_path


def _load_fraud_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if int(row.get("Class", 0)) == 1:
                rows.append(row)
    return rows


def _mutate_card_not_present(row: Dict) -> Dict:
    out = dict(row)
    amount = float(out.get("Amount", 0.0))
    out["Amount"] = max(1.0, amount * random.uniform(1.1, 2.2))
    out["V4"] = float(out.get("V4", 0.0)) + random.uniform(0.2, 1.2)
    out["V10"] = float(out.get("V10", 0.0)) + random.uniform(0.3, 1.5)
    out["fraud_category"] = "card_not_present"
    out["Class"] = 1
    return out


def _mutate_account_takeover(row: Dict) -> Dict:
    out = dict(row)
    out["Time"] = float(out.get("Time", 0.0)) + random.uniform(1.0, 120.0)
    out["V1"] = float(out.get("V1", 0.0)) + random.uniform(-1.5, 1.5)
    out["V2"] = float(out.get("V2", 0.0)) + random.uniform(-1.5, 1.5)
    out["V14"] = float(out.get("V14", 0.0)) + random.uniform(0.3, 1.6)
    out["fraud_category"] = "account_takeover"
    out["Class"] = 1
    return out


MUTATORS = {
    "card_not_present": _mutate_card_not_present,
    "account_takeover": _mutate_account_takeover,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment fraud dataset categories.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["card_not_present", "account_takeover"],
    )
    parser.add_argument("--target-count", type=int, default=200000)
    parser.add_argument("--synthetic-engine", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument(
        "--base-dataset",
        type=str,
        default="datasets/fraud_detection/synthetic_500k.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/fraud_detection/augmented_fraud_categories.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    base_path = Path(args.base_dataset)
    assert_allowed_training_path(base_path, context="fraud augmentation source")
    base_rows = _load_fraud_rows(base_path)
    if not base_rows:
        raise RuntimeError(f"No fraud rows found in {base_path}")

    output_rows: List[Dict] = []
    for category in args.categories:
        if category not in MUTATORS:
            raise ValueError(f"Unsupported category: {category}")
        mutator = MUTATORS[category]
        for _ in range(args.target_count):
            base = random.choice(base_rows)
            row = mutator(base)
            row["synthetic_engine"] = bool(args.synthetic_engine)
            output_rows.append(row)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(output_rows)} rows to {out_path}")
    by_category = {}
    for category in args.categories:
        by_category[category] = sum(1 for r in output_rows if r.get("fraud_category") == category)
    print("Category counts:", by_category)


if __name__ == "__main__":
    main()
