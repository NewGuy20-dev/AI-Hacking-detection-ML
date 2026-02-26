"""Tests for benign prefilter behavior."""
from src.prefilters.benign_pre_filter import BenignPreFilter


def test_safe_http_patterns_short_circuit():
    filt = BenignPreFilter()
    is_benign, conf, reason = filt.is_benign("GET /static/app.js")
    assert is_benign
    assert reason == "safe_pattern"
    assert conf >= 0.95


def test_attack_patterns_do_not_short_circuit():
    filt = BenignPreFilter()
    is_benign, conf, reason = filt.is_benign("<script>alert(1)</script>")
    assert not is_benign
    assert conf == 0.0
    assert reason is None


def test_predict_bypasses_cnn_for_obvious_benign():
    filt = BenignPreFilter()

    called = {"value": False}

    def _cnn(_payload):
        called["value"] = True
        return 1, 0.99

    pred, score = filt.predict("GET /assets/logo.svg", _cnn)
    assert pred == 0
    assert score < 0.1
    assert not called["value"]
