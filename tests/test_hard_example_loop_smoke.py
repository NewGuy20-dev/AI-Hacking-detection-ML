"""Smoke tests for feedback-loop orchestrator."""
import json
from argparse import Namespace
from pathlib import Path

from src.feedback_loop.hard_example_loop import run_loop


def test_loop_dry_run_writes_manifest_and_summary(tmp_path: Path):
    repo = tmp_path
    (repo / "config").mkdir(parents=True)
    (repo / "evaluation" / "stress_test_v14").mkdir(parents=True)
    (repo / "datasets" / "security_payloads" / "injection").mkdir(parents=True, exist_ok=True)
    (repo / "datasets" / "curated_benign").mkdir(parents=True, exist_ok=True)

    (repo / "config" / "feedback_loop.yaml").write_text("{}", encoding="utf-8")

    row = {
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
    }
    failure_file = repo / "evaluation" / "stress_test_v14" / "payload_2026-02-19_failures.jsonl"
    failure_file.write_text(json.dumps(row) + "\n", encoding="utf-8")

    args = Namespace(
        model="payload",
        run_date="2026-02-19",
        input_dir="evaluation/stress_test_v14",
        output_dir="evaluation/feedback_loop",
        models_dir="models",
        config="config/feedback_loop.yaml",
        repo_root=str(repo),
        max_failures_per_category=100,
        variants_per_failure=2,
        replay_ratio=0.6,
        hard_ratio_cap=0.4,
        baseline_max_samples=100,
        previous_max_samples=50,
        seed=42,
        dry_run=True,
        promote=False,
        training_timeout_seconds=60,
    )

    code = run_loop(args)
    assert code == 0

    runs = sorted((repo / "evaluation" / "feedback_loop").glob("*"))
    assert runs
    latest = runs[-1]
    assert (latest / "candidate_dataset_manifest.json").exists()
    assert (latest / "loop_summary.json").exists()
