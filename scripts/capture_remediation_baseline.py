#!/usr/bin/env python3
"""Capture a baseline snapshot from a stress-test run folder."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build remediation baseline from *_ops.json files.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="evaluation/stress_test_v14/2026-02-25",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/remediation_baseline_2026-02-25.json",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    baseline = {}
    for ops_file in sorted(input_dir.glob("*_ops.json")):
        model_name = ops_file.name.split("_")[0]
        baseline[model_name] = json.loads(ops_file.read_text(encoding="utf-8"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    print(f"Saved baseline snapshot for {len(baseline)} models to {output_path}")


if __name__ == "__main__":
    main()
