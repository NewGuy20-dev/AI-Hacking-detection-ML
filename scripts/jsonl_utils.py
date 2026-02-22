#!/usr/bin/env python3
"""Shared JSONL utilities."""
import json
from pathlib import Path


def iter_records(path: Path):
    with Path(path).open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
