#!/usr/bin/env python3
"""Aggregate 5 partial chunk results into final report_json payload."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate partial analysis results")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--partials-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    partial_files = sorted(args.partials_dir.glob("partial_*.json"))
    if len(partial_files) != 5:
        raise ValueError(f"Expected 5 partial files, found {len(partial_files)} in {args.partials_dir}")

    merged_move_data: list[dict[str, Any]] = []
    completed_chunks = 0
    for partial_file in partial_files:
        partial = json.loads(partial_file.read_text(encoding="utf-8"))
        merged_move_data.extend(partial.get("move_data", []))
        completed_chunks += 1

    report_json = {
        "job_id": args.job_id,
        "status": "done",
        "completed_at": _iso_now(),
        "summary": {
            "chunks_completed": completed_chunks,
            "games_analysed": len(merged_move_data),
        },
        "move_data": merged_move_data,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_json), encoding="utf-8")
    print(json.dumps(report_json["summary"]))


if __name__ == "__main__":
    main()
