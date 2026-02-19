"""Tests for feedback-loop failure ingestion."""
import json
from pathlib import Path

from src.feedback_loop.failure_ingest import ingest_failures


def test_ingest_failures_dedupes_and_parses(tmp_path: Path):
    log_dir = tmp_path / "evaluation" / "stress_test_v14"
    log_dir.mkdir(parents=True)

    path = log_dir / "payload_2026-02-19_failures.jsonl"
    rows = [
        {
            "scenario_id": "s1",
            "model": "payload",
            "category": "sqli",
            "subcategory": "dynamic",
            "expected": 1,
            "predicted": 0,
            "confidence": 0.1,
            "difficulty": "hard",
            "source": "dynamic",
            "tags": ["sqli"],
            "run_seed": 42,
            "input_preview": "' OR 1=1--",
            "timestamp": "2026-02-19T00:00:00Z",
            "error": None,
        },
        # duplicate by content should be removed
        {
            "scenario_id": "s2",
            "model": "payload",
            "category": "sqli",
            "subcategory": "dynamic",
            "expected": 1,
            "predicted": 0,
            "confidence": 0.2,
            "difficulty": "hard",
            "source": "dynamic",
            "tags": ["sqli"],
            "run_seed": 42,
            "input_preview": "' OR 1=1--",
            "timestamp": "2026-02-19T00:00:01Z",
            "error": None,
        },
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(rows[0]) + "\n")
        handle.write("not-json\n")
        handle.write(json.dumps(rows[1]) + "\n")

    out = ingest_failures(log_dir, models=["payload"], run_date="2026-02-19")
    assert out["total"] == 1
    assert out["records"][0].category == "sqli"
    assert out["source_files"]["payload"].endswith("payload_2026-02-19_failures.jsonl")
