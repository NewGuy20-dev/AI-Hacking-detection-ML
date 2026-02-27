"""Smoke tests for Hugging Face-hosted torch models."""

from tests.hf_validation.smoke_eval import run_smoke_validation


def test_hf_torch_models_smoke_forward_pass(hf_artifacts) -> None:
    """
    Run one deterministic forward pass per torch model.

    Assertions include:
    - model loads without crash
    - output reduces to one probability
    - probabilities are in [0,1]
    - meta classifier probability shape is exactly (1,1)
    """
    results = run_smoke_validation(hf_artifacts)
    assert set(results.keys()) == {
        "payload_cnn",
        "url_cnn",
        "timeseries_lstm",
        "meta_classifier",
    }

