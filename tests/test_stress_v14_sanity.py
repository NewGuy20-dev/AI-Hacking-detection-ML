import numpy as np

from src.stress_test.v14.ops_metrics import OpsMetricsState
from src.stress_test.v14.scenarios import MetaGenerator


def test_ops_metrics_sanity_flags():
    metrics = OpsMetricsState()
    # 10 perfect correct predictions: 5 benign, 5 malicious
    for _ in range(5):
        metrics.update(expected=0, predicted=0, confidence=0.1, latency_ms=1.0,
                      category="benign", difficulty="easy")
    for _ in range(5):
        metrics.update(expected=1, predicted=1, confidence=0.9, latency_ms=1.0,
                      category="attack", difficulty="easy")
    summary = metrics.summary()
    assert "perfect_accuracy_no_errors" in summary["sanity"]

    metrics = OpsMetricsState()
    for _ in range(5):
        metrics.update(expected=1, predicted=0, confidence=0.0, latency_ms=1.0,
                      category="attack", difficulty="easy")
    summary = metrics.summary()
    assert "zero_true_positives" in summary["sanity"]


def test_meta_generator_overlap_adversarial():
    gen = MetaGenerator(seed=42)
    benign = []
    malicious = []
    for _ in range(500):
        benign.append(gen._sample_meta_vector(label=0, difficulty="adversarial").mean())
        malicious.append(gen._sample_meta_vector(label=1, difficulty="adversarial").mean())

    benign = np.array(benign)
    malicious = np.array(malicious)
    # Overlap should exist in adversarial distribution
    assert benign.max() > malicious.min()
