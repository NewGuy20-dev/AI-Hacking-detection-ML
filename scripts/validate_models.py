#!/usr/bin/env python3
"""Compatibility wrapper for src/validate.py."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src/validate.py"

if not TARGET.exists():
    raise SystemExit(f"Target script not found: {TARGET}")

sys.path.insert(0, str(ROOT))
runpy.run_path(str(TARGET), run_name="__main__")
