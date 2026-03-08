import numpy as np

from src.stress_test.v14.ops_metrics import OpsMetricsState
from src.stress_test.v14.runner import StressTestRunner
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


def test_timeseries_preflight_rejects_collapsed_confidence():
    class FakeModel:
        def predict(self, _input):
            return 0, 0.0, 1.0

        def consume_last_prediction_metadata(self):
            return {}

    runner = StressTestRunner("timeseries", {"target_duration_min": 1})
    scenarios = [
        type("ScenarioStub", (), {"id": "benign", "expected_label": 0, "input_data": [], "category": "normal", "difficulty": "easy"})(),
        type("ScenarioStub", (), {"id": "attack", "expected_label": 1, "input_data": [], "category": "ddos", "difficulty": "easy"})(),
    ]

    try:
        runner._run_preflight(FakeModel(), scenarios)
    except RuntimeError as exc:
        assert "collapsed timeseries confidence distribution" in str(exc)
    else:
        raise AssertionError("Expected collapsed timeseries preflight to fail")
