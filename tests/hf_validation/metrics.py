"""Metrics and regression guardrail helpers for model validation tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class BenchmarkResult:
    """Computed benchmark metrics for one model."""

    accuracy: float
    recall: float
    fpr: float
    avg_latency_ms: float
    n_samples: int


@dataclass(frozen=True)
class RegressionViolation:
    """One baseline threshold violation."""

    model: str
    metric: str
    actual: float
    bound: float
    rule: str


def compute_binary_metrics(labels: list[int], predictions: list[int]) -> tuple[float, float, float]:
    """Return accuracy, recall, and false positive rate."""
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same length.")
    if not labels:
        raise ValueError("Cannot compute metrics on empty labels.")

    tp = fp = tn = fn = 0
    for label, pred in zip(labels, predictions):
        if label == 1 and pred == 1:
            tp += 1
        elif label == 0 and pred == 1:
            fp += 1
        elif label == 0 and pred == 0:
            tn += 1
        elif label == 1 and pred == 0:
            fn += 1

    total = len(labels)
    accuracy = (tp + tn) / total
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return accuracy, recall, fpr


def evaluate_against_baseline(
    results: Mapping[str, BenchmarkResult],
    baselines: Mapping[str, Mapping[str, float]],
) -> list[RegressionViolation]:
    """Compare benchmark results against baseline bounds."""
    violations: list[RegressionViolation] = []
    for model_name, result in results.items():
        if model_name not in baselines:
            raise ValueError(f"Missing baseline config for model '{model_name}'.")
        baseline = baselines[model_name]

        min_accuracy = float(baseline["min_accuracy"])
        min_recall = float(baseline["min_recall"])
        max_fpr = float(baseline["max_fpr"])
        max_latency = float(baseline["max_latency_ms"])

        if result.accuracy < min_accuracy:
            violations.append(
                RegressionViolation(
                    model=model_name,
                    metric="accuracy",
                    actual=result.accuracy,
                    bound=min_accuracy,
                    rule="min",
                )
            )
        if result.recall < min_recall:
            violations.append(
                RegressionViolation(
                    model=model_name,
                    metric="recall",
                    actual=result.recall,
                    bound=min_recall,
                    rule="min",
                )
            )
        if result.fpr > max_fpr:
            violations.append(
                RegressionViolation(
                    model=model_name,
                    metric="fpr",
                    actual=result.fpr,
                    bound=max_fpr,
                    rule="max",
                )
            )
        if result.avg_latency_ms > max_latency:
            violations.append(
                RegressionViolation(
                    model=model_name,
                    metric="avg_latency_ms",
                    actual=result.avg_latency_ms,
                    bound=max_latency,
                    rule="max",
                )
            )
    return violations

