"""Tests for V1.4 stress-test reliability and dynamic scenario behavior."""
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from src.stress_test.v14.models import ModelWrapper
from src.stress_test.v14.runner import AdaptiveScheduler, StressTestRunner
from src.stress_test.v14.scenarios import (
    Scenario,
    ScenarioResult,
    ScenarioRegistry,
    URLGenerator,
    TimeSeriesGenerator,
    TabularGenerator,
)
from src.stress_test.v14.difficulty import DifficultyMixin
from src.stress_test.v14.logger import JSONLogger


def test_runner_filters_non_malicious_categories():
    """Adaptive weighting input should exclude benign/normal categories."""
    runner = StressTestRunner('payload', {'target_duration_min': 1})
    filtered = runner._filter_malicious_accuracy({
        'sqli': 0.7,
        'xss': 0.6,
        'benign': 0.0,
        'normal': 1.0,
    })
    assert 'sqli' in filtered
    assert 'xss' in filtered
    assert 'benign' not in filtered
    assert 'normal' not in filtered


def test_adaptive_scheduler_normalizes():
    scheduler = AdaptiveScheduler({'a': 0.5, 'b': 0.5})
    weights = scheduler.compute_weights({'a': 0.1, 'b': 0.9})
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert set(weights.keys()) == {'a', 'b'}


def test_adaptive_scheduler_boosts_low_accuracy_category():
    scheduler = AdaptiveScheduler({'a': 0.5, 'b': 0.5})
    weights = scheduler.compute_weights({'a': 0.2, 'b': 0.9})
    assert weights['a'] > weights['b']


def test_url_generator_malicious_categories_never_use_benign_weight_key():
    """Generator should ignore unsupported scheduler keys for malicious sampling."""
    gen = URLGenerator(seed=7)
    scenarios = gen.generate(20, {'benign': 1.0}, benign_ratio=0.0)
    malicious_categories = {s.category for s in scenarios if s.expected_label == 1}
    assert malicious_categories
    assert 'benign' not in malicious_categories
    assert malicious_categories.issubset(set(URLGenerator.DEFAULT_MALICIOUS_WEIGHTS.keys()))


def test_url_generator_realism_includes_paths_queries_or_ports():
    gen = URLGenerator(seed=9)
    scenarios = gen.generate(30, {'phishing': 1.0}, benign_ratio=0.0)
    urls = [s.input_data for s in scenarios]
    assert all(u.startswith("http://") or u.startswith("https://") for u in urls)
    assert any("?" in u or "/" in u[8:] or ":" in u.split("//", 1)[1] for u in urls)


def test_url_difficulty_obfuscation_stays_parseable():
    mixin = DifficultyMixin()
    src_url = "http://paypal-login.com/account/verify?session=123"
    for difficulty in ['easy', 'medium', 'hard', 'adversarial']:
        candidate = mixin.apply_difficulty(src_url, difficulty, 'url')
        parsed = urlsplit(candidate)
        assert parsed.scheme in {'http', 'https'}
        assert parsed.netloc


def test_timeseries_generator_deterministic_with_seed():
    """Same seed should replay the same first scenario."""
    g1 = TimeSeriesGenerator(seed=123)
    s1 = g1.generate(3, {'ddos': 1.0}, benign_ratio=0.0)
    g2 = TimeSeriesGenerator(seed=123)
    s2 = g2.generate(3, {'ddos': 1.0}, benign_ratio=0.0)
    assert s1[0].category == s2[0].category
    assert s1[0].difficulty == s2[0].difficulty
    assert np.allclose(s1[0].input_data, s2[0].input_data)


def test_timeseries_generator_applies_difficulty_and_clipping():
    gen = TimeSeriesGenerator(seed=44)
    scenarios = gen.generate(10, {'exfiltration': 1.0}, benign_ratio=0.0)
    assert all(s.difficulty in {'easy', 'medium', 'hard', 'adversarial'} for s in scenarios)
    for s in scenarios:
        arr = np.asarray(s.input_data)
        assert arr.shape == (60, 8)
        assert np.min(arr) >= 0.0
        assert np.max(arr) <= 50000.0


def test_tabular_generator_anomaly_shape_and_labels():
    gen = TabularGenerator(seed=11)
    scenarios = gen.generate('anomaly', 8, {'zero_day': 1.0}, benign_ratio=0.0)
    assert len(scenarios) == 8
    assert all(s.expected_label == 1 for s in scenarios)
    assert all(s.category == 'zero_day' for s in scenarios)
    assert all(np.asarray(s.input_data).shape == (15,) for s in scenarios)


def test_tabular_generator_assigns_difficulty_to_malicious_samples():
    gen = TabularGenerator(seed=55)
    scenarios = gen.generate('network', 20, {'dos': 1.0}, benign_ratio=0.0)
    difficulties = {s.difficulty for s in scenarios}
    assert difficulties.issubset({'easy', 'medium', 'hard', 'adversarial'})
    assert len(difficulties) >= 2
    assert all(s.expected_label == 1 for s in scenarios)


def test_tabular_generator_creates_benign_hard_negatives():
    gen = TabularGenerator(seed=77)
    scenarios = gen.generate('fraud', 30, {'card_not_present': 1.0}, benign_ratio=1.0)
    assert all(s.expected_label == 0 for s in scenarios)
    assert any(s.difficulty in {'hard', 'adversarial'} for s in scenarios)


def test_model_wrapper_anomaly_predict_path_without_disk_load():
    """Anomaly wrapper should convert score/pred outputs to attack + confidence."""

    class FakeIsolationModel:
        def predict(self, X):
            return np.array([-1])  # anomaly

        def score_samples(self, X):
            return np.array([-0.9])  # larger negative => more anomalous

    wrapper = ModelWrapper('anomaly', models_dir=Path('models'))
    wrapper.model = FakeIsolationModel()
    wrapper.scaler = None

    pred, conf, latency = wrapper.predict(np.ones(15, dtype=np.float32))
    assert pred == 1
    assert 0.0 <= conf <= 1.0
    assert latency >= 0.0


def test_json_logger_serializes_numpy_scalar_fields(tmp_path):
    """Logger must serialize numpy scalar booleans/ints without crashing."""
    scenario = Scenario(
        id="network_mal_1",
        model="network",
        category="dos",
        subcategory="dynamic",
        input_data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        expected_label=1,
        difficulty="hard",
        description="test",
        source="dynamic",
    )
    result = ScenarioResult(
        scenario=scenario,
        prediction=int(np.int64(1)),
        confidence=float(np.float32(0.91)),
        passed=np.bool_(True),
        latency_ms=float(np.float32(1.25)),
        timestamp="2026-02-27T00:00:00",
        error=None,
    )

    with JSONLogger(tmp_path, "network", "2026-02-27", run_seed=10) as logger:
        logger.log(result)

    output_path = tmp_path / "2026-02-27" / "network_2026-02-27.jsonl"
    line = output_path.read_text(encoding="utf-8").strip()
    assert '"passed": true' in line


def test_scenario_registry_loads_file_backed_static_fixture(tmp_path):
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "ts.json").write_text(
        "[[1,2,3,4,5,6,7,8],[2,3,4,5,6,7,8,9]]",
        encoding="utf-8",
    )
    (tmp_path / "timeseries.yaml").write_text(
        "\n".join(
            [
                "scenarios:",
                "  - id: ts_fixture_001",
                "    category: ddos",
                "    subcategory: fixture",
                "    input_file: fixtures/ts.json",
                "    expected: 1",
                "    difficulty: medium",
                "    description: fixture",
            ]
        ),
        encoding="utf-8",
    )

    registry = ScenarioRegistry(tmp_path)
    scenarios = registry.load_static("timeseries")
    assert len(scenarios) == 1
    assert isinstance(scenarios[0].input_data, np.ndarray)
    assert scenarios[0].input_data.shape == (2, 8)
