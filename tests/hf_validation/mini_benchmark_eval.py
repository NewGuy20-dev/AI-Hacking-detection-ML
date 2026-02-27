"""Deterministic mini benchmark evaluation for HF model regression checks."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from tests.hf_validation.metrics import BenchmarkResult, compute_binary_metrics

BENCHMARK_MODELS: tuple[str, ...] = (
    "payload_cnn",
    "url_cnn",
    "timeseries_lstm",
    "meta_classifier",
)


def load_mini_benchmark(path: Path) -> dict[str, dict[str, Any]]:
    """Load compressed benchmark dataset from json.gz."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark file at {path} must contain JSON object.")
    for model_name in BENCHMARK_MODELS:
        model_entry = payload.get(model_name)
        if not isinstance(model_entry, dict):
            raise ValueError(f"Benchmark missing model entry: {model_name}")
        inputs = model_entry.get("inputs")
        labels = model_entry.get("labels")
        if not isinstance(inputs, list) or not isinstance(labels, list):
            raise ValueError(f"Benchmark entry {model_name} missing inputs/labels lists.")
        if len(inputs) != len(labels):
            raise ValueError(f"Benchmark entry {model_name} has input/label length mismatch.")
    return payload


def load_baseline_metrics(path: Path) -> dict[str, dict[str, float]]:
    """Load baseline threshold config for regression checks."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Baseline metrics file at {path} must contain JSON object.")
    return payload


def _to_input_tensor(model_name: str, sample: Any) -> torch.Tensor:
    if model_name in ("payload_cnn", "url_cnn"):
        return torch.tensor(sample, dtype=torch.long).unsqueeze(0)
    if model_name == "timeseries_lstm":
        return torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
    if model_name == "meta_classifier":
        return torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
    raise ValueError(f"Unsupported model for benchmark conversion: {model_name}")


def run_model_benchmark(
    *,
    model_name: str,
    model: torch.nn.Module,
    threshold: float,
    samples: list[Any],
    labels: list[int],
) -> BenchmarkResult:
    """Run deterministic per-sample inference and compute regression metrics."""
    if len(samples) != len(labels):
        raise ValueError(f"Sample/label mismatch for {model_name}")
    if not samples:
        raise ValueError(f"No benchmark samples provided for {model_name}")

    model.eval()
    predictions: list[int] = []
    latencies_ms: list[float] = []

    warmup = _to_input_tensor(model_name, samples[0])
    with torch.inference_mode():
        _ = model(warmup)

        for sample in samples:
            inputs = _to_input_tensor(model_name, sample)
            started = time.perf_counter()
            logits = model(inputs)
            probs = torch.sigmoid(logits.reshape(1, -1))
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            probability = float(probs.item())
            prediction = 1 if probability >= threshold else 0
            predictions.append(prediction)
            latencies_ms.append(elapsed_ms)

    accuracy, recall, fpr = compute_binary_metrics(labels, predictions)
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)
    return BenchmarkResult(
        accuracy=accuracy,
        recall=recall,
        fpr=fpr,
        avg_latency_ms=avg_latency_ms,
        n_samples=len(labels),
    )


def run_mini_benchmark_suite(
    *,
    loaded_models: Mapping[str, tuple[torch.nn.Module, float]],
    dataset: Mapping[str, Mapping[str, Any]],
) -> dict[str, BenchmarkResult]:
    """Run mini benchmark for all gate models."""
    results: dict[str, BenchmarkResult] = {}
    for model_name in BENCHMARK_MODELS:
        if model_name not in loaded_models:
            raise ValueError(f"Loaded models missing '{model_name}'.")
        model, threshold = loaded_models[model_name]
        entry = dataset[model_name]
        samples = entry["inputs"]
        labels = [int(x) for x in entry["labels"]]
        results[model_name] = run_model_benchmark(
            model_name=model_name,
            model=model,
            threshold=float(threshold),
            samples=samples,
            labels=labels,
        )
    return results

