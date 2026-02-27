"""Deterministic mini benchmark regression tests for HF-hosted torch models."""

from __future__ import annotations

import json
from pathlib import Path

from tests.hf_validation.metrics import evaluate_against_baseline
from tests.hf_validation.mini_benchmark_eval import (
    load_baseline_metrics,
    load_mini_benchmark,
    run_mini_benchmark_suite,
)
from tests.hf_validation.smoke_eval import load_torch_models_for_eval


def test_hf_model_regression_guardrails(hf_artifacts) -> None:
    """Fail CI when benchmark metrics regress beyond configured baseline bounds."""
    benchmark_path = Path("tests/mini_benchmark/data.json.gz")
    baseline_path = Path("tests/mini_benchmark/baseline_metrics.json")

    dataset = load_mini_benchmark(benchmark_path)
    baselines = load_baseline_metrics(baseline_path)
    loaded_models = load_torch_models_for_eval(hf_artifacts)

    results = run_mini_benchmark_suite(loaded_models=loaded_models, dataset=dataset)
    violations = evaluate_against_baseline(results=results, baselines=baselines)

    print(
        "Mini benchmark results:\n"
        + json.dumps(
            {
                model: {
                    "accuracy": metrics.accuracy,
                    "recall": metrics.recall,
                    "fpr": metrics.fpr,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "n_samples": metrics.n_samples,
                }
                for model, metrics in results.items()
            },
            indent=2,
            sort_keys=True,
        )
    )

    assert not violations, (
        "Model regression baseline violation(s):\n"
        + "\n".join(
            f"- {v.model}.{v.metric}: actual={v.actual:.6f}, "
            f"{'minimum' if v.rule == 'min' else 'maximum'}={v.bound:.6f}"
            for v in violations
        )
    )

