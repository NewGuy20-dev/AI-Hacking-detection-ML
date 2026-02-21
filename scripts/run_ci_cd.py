#!/usr/bin/env python3
"""Run CI/CD validation checks locally (Windows and Unix-like shells)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _npm_executable() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def _run_step(name: str, cmd: Sequence[str], cwd: Path | None = None) -> Tuple[str, int, float]:
    start = time.time()
    shown_cwd = str(cwd or ROOT)
    print(f"\n=== {name} ===")
    print(f"CWD: {shown_cwd}")
    print(f"CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd or ROOT), check=False)
    duration = time.time() - start
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"RESULT: {status} ({duration:.1f}s)")
    return name, result.returncode, duration


def build_steps(args: argparse.Namespace) -> List[Tuple[str, List[str], Path | None]]:
    steps: List[Tuple[str, List[str], Path | None]] = []
    py = sys.executable

    if not args.skip_tests:
        test_cmd = [
            py,
            "-m",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--ignore=tests/test_robust_training.py",
        ]
        if args.pytest_args:
            test_cmd.extend(args.pytest_args)
        steps.append(("Pytest", test_cmd, ROOT))

    if not args.skip_stress_smoke:
        steps.append(
            ("Stress V1.4 Smoke", [py, "scripts/test_adversarial_enhancements.py"], ROOT)
        )

    if not args.skip_feedback_smoke:
        steps.append(
            (
                "Feedback Loop Smoke",
                [
                    py,
                    "src/feedback_loop/hard_example_loop.py",
                    "--model",
                    "payload,url",
                    "--dry-run",
                ],
                ROOT,
            )
        )

    if args.with_dashboard_lint:
        steps.append(("Dashboard Lint", [_npm_executable(), "run", "lint"], ROOT / "dashboard"))

    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local CI/CD checks (Windows-compatible)."
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest step.",
    )
    parser.add_argument(
        "--skip-stress-smoke",
        action="store_true",
        help="Skip scripts/test_adversarial_enhancements.py.",
    )
    parser.add_argument(
        "--skip-feedback-smoke",
        action="store_true",
        help="Skip hard_example_loop dry-run smoke test.",
    )
    parser.add_argument(
        "--with-dashboard-lint",
        action="store_true",
        help="Also run dashboard lint (npm run lint).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining steps even after a failure.",
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to pytest (example: --pytest-args -k api -x).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = build_steps(args)
    if not steps:
        print("No steps selected. Nothing to run.")
        return 0

    print("Running local CI/CD checks")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.executable}")

    results: List[Tuple[str, int, float]] = []
    for name, cmd, cwd in steps:
        step = _run_step(name, cmd, cwd)
        results.append(step)
        if step[1] != 0 and not args.continue_on_error:
            break

    print("\n=== Summary ===")
    failures = 0
    for name, code, duration in results:
        ok = code == 0
        if not ok:
            failures += 1
        status = "PASS" if ok else f"FAIL({code})"
        print(f"{status:10s} {name:24s} {duration:6.1f}s")

    if failures:
        print(f"\nCompleted with {failures} failing step(s).")
        return 1

    print("\nAll selected CI/CD steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
