"""Tests for strict promotion gating."""

from src.feedback_loop.gating import GatingThresholds, evaluate_gates


def test_gating_passes_when_thresholds_met():
    baseline = {"targeted_recall": 0.80, "fpr": 0.02, "p95_latency_ms": 100.0}
    candidate = {"targeted_recall": 0.83, "fpr": 0.022, "p95_latency_ms": 108.0}
    thresholds = GatingThresholds(
        min_targeted_recall_delta=0.02,
        max_fpr_regression=0.005,
        max_latency_regression_pct=0.10,
    )

    result = evaluate_gates(baseline, candidate, thresholds)
    assert result["passed"] is True
    assert all(v["passed"] for v in result["gates"].values())


def test_gating_fails_on_fpr_regression():
    baseline = {"targeted_recall": 0.80, "fpr": 0.01, "p95_latency_ms": 100.0}
    candidate = {"targeted_recall": 0.85, "fpr": 0.03, "p95_latency_ms": 102.0}
    thresholds = GatingThresholds(
        min_targeted_recall_delta=0.02,
        max_fpr_regression=0.005,
        max_latency_regression_pct=0.10,
    )

    result = evaluate_gates(baseline, candidate, thresholds)
    assert result["passed"] is False
    assert result["gates"]["fpr"]["passed"] is False
