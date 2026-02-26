#!/usr/bin/env python3
"""Profile slow network intrusion inferences from stress-test logs."""
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze slow network inference scenarios.")
    parser.add_argument(
        "--input-log",
        type=str,
        default="evaluation/stress_test_v14/2026-02-25/network_2026-02-25.jsonl",
    )
    parser.add_argument("--slow-ms", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.input_log)
    if not path.exists():
        raise FileNotFoundError(f"Missing log file: {path}")

    by_diff = Counter()
    by_cat = Counter()
    slow_rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            latency = float(row["latency_ms"])
            if latency >= args.slow_ms:
                by_diff[row.get("difficulty", "unknown")] += 1
                by_cat[row.get("category", "unknown")] += 1
                slow_rows.append(row)

    slow_rows.sort(key=lambda r: float(r["latency_ms"]), reverse=True)

    print(f"Input log      : {path}")
    print(f"Slow threshold : {args.slow_ms} ms")
    print(f"Slow count     : {len(slow_rows)}")
    print("By difficulty  :", dict(by_diff))
    print("By category    :", dict(by_cat))
    print("\nTop slow samples:")
    for row in slow_rows[: args.top_k]:
        item = {
            "scenario_id": row.get("scenario_id"),
            "difficulty": row.get("difficulty"),
            "category": row.get("category"),
            "latency_ms": row.get("latency_ms"),
            "input_preview": row.get("input_preview"),
        }
        if "input_summary" in row:
            item["input_summary"] = row["input_summary"]
        print(item)

    if slow_rows and "input_summary" not in slow_rows[0]:
        print(
            "\nNote: input_summary is missing in this run. "
            "Set STRESS_LOG_INPUT_SUMMARY=1 before stress test to capture feature stats."
        )


if __name__ == "__main__":
    main()
