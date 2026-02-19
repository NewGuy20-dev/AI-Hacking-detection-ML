"""Strict gating for model promotion in feedback loop."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class GatingThresholds:
    min_targeted_recall_delta: float = 0.02
    max_fpr_regression: float = 0.005
    max_latency_regression_pct: float = 0.10


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def evaluate_model_metrics(models_dir: str | Path, model: str, eval_samples: List[Dict]) -> Dict:
    """Evaluate simple recall/FPR/latency for a model on eval samples."""
    from src.hybrid_predictor import HybridPredictor

    predictor = HybridPredictor(str(models_dir))
    predictor.load_models()

    malicious_total = 0
    malicious_tp = 0
    benign_total = 0
    benign_fp = 0
    latencies_ms: List[float] = []
    by_category: Dict[str, Dict[str, int]] = {}

    for sample in eval_samples:
        text = sample.get("text", "")
        expected = int(sample.get("label", 0))
        category = sample.get("category", "unknown")

        start = time.perf_counter()
        if model == "payload":
            score = float(predictor.predict_payload([text])[0])
        elif model == "url":
            score = float(predictor.predict_url([text])[0])
        else:
            raise ValueError(f"Unsupported model for gating: {model}")
        latency_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(latency_ms)

        predicted = 1 if score > 0.5 else 0
        bucket = by_category.setdefault(category, {"total": 0, "tp": 0, "fp": 0, "malicious": 0, "benign": 0})
        bucket["total"] += 1

        if expected == 1:
            malicious_total += 1
            bucket["malicious"] += 1
            if predicted == 1:
                malicious_tp += 1
                bucket["tp"] += 1
        else:
            benign_total += 1
            bucket["benign"] += 1
            if predicted == 1:
                benign_fp += 1
                bucket["fp"] += 1

    latencies_ms.sort()
    p95_latency = latencies_ms[int(0.95 * (len(latencies_ms) - 1))] if latencies_ms else 0.0

    return {
        "model": model,
        "samples": len(eval_samples),
        "targeted_recall": _safe_div(malicious_tp, malicious_total),
        "fpr": _safe_div(benign_fp, benign_total),
        "p95_latency_ms": p95_latency,
        "malicious_total": malicious_total,
        "benign_total": benign_total,
        "by_category": by_category,
    }


def evaluate_gates(
    baseline: Dict,
    candidate: Dict,
    thresholds: GatingThresholds,
) -> Dict:
    """Evaluate strict promotion gates using baseline and candidate metrics."""
    recall_delta = candidate.get("targeted_recall", 0.0) - baseline.get("targeted_recall", 0.0)
    fpr_delta = candidate.get("fpr", 0.0) - baseline.get("fpr", 0.0)

    base_latency = baseline.get("p95_latency_ms", 0.0)
    cand_latency = candidate.get("p95_latency_ms", 0.0)
    latency_delta_pct = ((cand_latency - base_latency) / base_latency) if base_latency > 0 else 0.0

    gates = {
        "targeted_recall": {
            "passed": recall_delta >= thresholds.min_targeted_recall_delta,
            "baseline": baseline.get("targeted_recall", 0.0),
            "candidate": candidate.get("targeted_recall", 0.0),
            "delta": recall_delta,
            "threshold": thresholds.min_targeted_recall_delta,
        },
        "fpr": {
            "passed": fpr_delta <= thresholds.max_fpr_regression,
            "baseline": baseline.get("fpr", 0.0),
            "candidate": candidate.get("fpr", 0.0),
            "delta": fpr_delta,
            "threshold": thresholds.max_fpr_regression,
        },
        "latency": {
            "passed": latency_delta_pct <= thresholds.max_latency_regression_pct,
            "baseline_ms": base_latency,
            "candidate_ms": cand_latency,
            "delta_pct": latency_delta_pct,
            "threshold": thresholds.max_latency_regression_pct,
        },
    }

    return {
        "passed": all(g["passed"] for g in gates.values()),
        "gates": gates,
        "baseline": baseline,
        "candidate": candidate,
    }
