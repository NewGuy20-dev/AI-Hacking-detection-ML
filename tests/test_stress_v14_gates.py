"""Unit tests for V1.4 gate evaluation."""

import sys
from pathlib import Path

# Allow `python tests/test_stress_v14_gates.py` from repo root on Windows/WSL.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.stress_test.v14.gates import GateEvaluator


def _payload_like_ops():
    return {
        "metrics": {
            "recall": 0.94,
            "fp_rate": 0.02,
            "ece": 0.1,
        },
        "latency": {
            "p95_ms": 4.0,
        },
        "per_category": {
            "sqli": {"tp": 95, "fn": 5, "fp": 0, "tn": 0},
            "xss": {"tp": 90, "fn": 10, "fp": 0, "tn": 0},
            "benign": {"tp": 0, "fn": 0, "fp": 2, "tn": 98},
        },
        "per_difficulty": {
            "adversarial": {"tp": 20, "fn": 2, "fp": 1, "tn": 29},
        },
        "sanity": [],
    }


def test_gate_evaluator_passes_happy_path():
    evaluator = GateEvaluator.from_path("config/stress_test/gates_v14.yaml")
    ops = _payload_like_ops()
    report = evaluator.evaluate(
        model_name="payload",
        ops=ops,
        static_count=2,
        run_seed=42,
        fail_on_sanity=True,
    )
    assert report["passed"] is True
    assert all(check["passed"] for check in report["checks"])


def test_gate_evaluator_enforces_critical_sanity_flags():
    evaluator = GateEvaluator.from_path("config/stress_test/gates_v14.yaml")
    ops = _payload_like_ops()
    ops["sanity"] = ["zero_true_positives"]

    strict = evaluator.evaluate(
        model_name="payload",
        ops=ops,
        static_count=2,
        run_seed=42,
        fail_on_sanity=True,
    )
    permissive = evaluator.evaluate(
        model_name="payload",
        ops=ops,
        static_count=2,
        run_seed=42,
        fail_on_sanity=False,
    )

    strict_check = next(c for c in strict["checks"] if c["id"] == "critical_sanity_flags")
    permissive_check = next(c for c in permissive["checks"] if c["id"] == "critical_sanity_flags")
    assert strict_check["passed"] is False
    assert permissive_check["passed"] is True
    assert strict["passed"] is False


def test_gate_evaluator_requires_seed_and_static_fixtures():
    evaluator = GateEvaluator.from_path("config/stress_test/gates_v14.yaml")
    ops = _payload_like_ops()
    report = evaluator.evaluate(
        model_name="payload",
        ops=ops,
        static_count=0,
        run_seed=None,
        fail_on_sanity=True,
    )
    failed = {check["id"] for check in report["checks"] if not check["passed"]}
    assert "run_seed_present" in failed
    assert "static_fixture_count" in failed
