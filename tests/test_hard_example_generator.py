"""Tests for hard-example generation."""

from src.feedback_loop.failure_ingest import FailureRecord
from src.feedback_loop.hard_example_generator import HardExampleGenerator


def _record(model: str, category: str, expected: int, preview: str):
    return FailureRecord(
        scenario_id="x1",
        model=model,
        category=category,
        subcategory="dynamic",
        expected=expected,
        predicted=1 - expected,
        confidence=0.2,
        difficulty="hard",
        source="dynamic",
        tags=[category],
        run_seed=42,
        input_preview=preview,
        timestamp="2026-02-19T00:00:00Z",
        error=None,
        record_hash="abc123" + model,
    )


def test_generator_outputs_payload_and_url_variants():
    failures = [
        _record("payload", "sqli", 1, "' OR 1=1--"),
        _record("url", "phishing", 1, "http://paypa1.com/login"),
    ]

    gen = HardExampleGenerator(seed=7, variants_per_failure=3)
    out = gen.generate(failures)

    assert len(out) >= 4
    assert {x["model"] for x in out} == {"payload", "url"}
    assert all("text" in x and x["text"] for x in out)
    assert all(x["origin"] == "failure_loop" for x in out)
