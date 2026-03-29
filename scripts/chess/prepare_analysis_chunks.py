#!/usr/bin/env python3
"""Prepare 5 fixed-size analysis chunks from a 300-game source file.

Input format: JSON array with game objects or JSONL (one game object per line).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_games(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Input file is empty: {path}")

    if raw[0] == "[":
        games = json.loads(raw)
        if not isinstance(games, list):
            raise ValueError("JSON input must be an array of games")
        return games

    games = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return games


def _write_chunk(chunk: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split 300 games into five chunks of 60")
    parser.add_argument("--input", required=True, type=Path, help="Path to preprocessed games JSON/JSONL")
    parser.add_argument("--output-dir", required=True, type=Path, help="Chunk output directory")
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--chunks", type=int, default=5)
    args = parser.parse_args()

    games = _load_games(args.input)
    expected_games = args.chunk_size * args.chunks
    if len(games) != expected_games:
        raise ValueError(
            f"Expected exactly {expected_games} games for fan-out; found {len(games)} in {args.input}"
        )

    for chunk_id in range(args.chunks):
        start = chunk_id * args.chunk_size
        end = start + args.chunk_size
        chunk = games[start:end]
        _write_chunk(chunk, args.output_dir / f"chunk_{chunk_id}.json")

    manifest = {
        "chunks": args.chunks,
        "chunk_size": args.chunk_size,
        "total_games": len(games),
        "chunk_files": [f"chunk_{i}.json" for i in range(args.chunks)],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
