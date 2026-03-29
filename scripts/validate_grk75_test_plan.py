#!/usr/bin/env python3
"""Validate completeness/consistency of docs/GRK-75_E2E_TEST_PLAN.md."""

from __future__ import annotations

from pathlib import Path
import re
import sys

PLAN_PATH = Path("docs/GRK-75_E2E_TEST_PLAN.md")

REQUIRED_MATRIX_IDS = [
    "P1", "P2", "P3", "P4", "P5", "P6", "P7",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "R1", "R2", "R3",
]

REQUIRED_KEY_PHRASES = [
    "300-game",
    "report_json",
    "analysis_partial_results",
    "X/5 chunks done",
    "10–11",
    "WDL",
    "pv[0]",
    "Chess.com",
    "move 15",
    "Brilliant",
    "exactly one label",
    "analysis_cache",
]


def validate() -> list[str]:
    errors: list[str] = []

    if not PLAN_PATH.exists():
        return [f"Missing file: {PLAN_PATH}"]

    text = PLAN_PATH.read_text(encoding="utf-8")

    # Validate matrix IDs are present in the table rows.
    for case_id in REQUIRED_MATRIX_IDS:
        if f"| {case_id} |" not in text:
            errors.append(f"Matrix row for {case_id} is missing")

    # Validate expected key phrases requested by issue context.
    for phrase in REQUIRED_KEY_PHRASES:
        if phrase not in text:
            errors.append(f"Required phrase is missing: {phrase!r}")

    # Basic heading nesting consistency checks.
    if "## Test Data & Environment Prerequisites\n\n## Environment" in text:
        errors.append("Heading level mismatch: Environment should be nested under prerequisites")
    if "## Execution Steps\n\n## Phase A" in text:
        errors.append("Heading level mismatch: Phase A should be nested under execution steps")

    # Ensure each scenario section has pass conditions.
    for sec in [
        "A1", "A2", "A3", "A4", "A5", "A6", "A7",
        "B1", "B2", "B3", "B4", "B5", "B6", "B7",
        "C1", "C2", "C3",
    ]:
        pattern = rf"### {sec}\..*?(\*\*Pass conditions\*\*)"
        if not re.search(pattern, text, flags=re.DOTALL):
            errors.append(f"Missing pass conditions block for section {sec}")

    return errors


def main() -> int:
    errors = validate()
    if errors:
        print("GRK-75 test-plan validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("GRK-75 test-plan validation passed.")
    print(f"Validated file: {PLAN_PATH}")
    print(f"Scenarios checked: {len(REQUIRED_MATRIX_IDS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
