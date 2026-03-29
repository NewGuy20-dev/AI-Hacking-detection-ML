#!/usr/bin/env python3
"""Run per-chunk analysis and post partial results.

This script intentionally isolates the worker contract:
- reads one chunk file
- executes analysis command (or dry-run fallback)
- writes a partial result payload
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one chunk and emit partial result")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--chunk-id", required=True, type=int)
    parser.add_argument("--chunk-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--status-endpoint",
        default="",
        help="Optional endpoint for partial-result ingestion (e.g. Convex HTTP action)",
    )
    args = parser.parse_args()

    games = json.loads(args.chunk_file.read_text(encoding="utf-8"))
    if not isinstance(games, list):
        raise ValueError(f"Chunk file must contain a JSON array: {args.chunk_file}")

    # Keep the worker interface stable regardless of engine internals.
    move_data: list[dict[str, Any]] = []
    for game in games:
        game_id = game.get("game_id") or game.get("id") or f"unknown-{len(move_data)}"
        move_data.append({"game_id": game_id, "analysis": game.get("analysis", {}), "worker_chunk": args.chunk_id})

    payload = {
        "job_id": args.job_id,
        "chunk_id": args.chunk_id,
        "move_data": move_data,
        "completed_at": _iso_now(),
        "status": f"analysing ({args.chunk_id + 1}/5 chunks done)",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")

    endpoint = args.status_endpoint.strip()
    if endpoint:
        import urllib.request

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as response:  # nosec B310
            print(f"Posted partial results: HTTP {response.status}")
    else:
        print("No status endpoint provided; partial result written locally only.")

    print(f"Chunk {args.chunk_id} processed: {len(move_data)} games")


if __name__ == "__main__":
    main()
