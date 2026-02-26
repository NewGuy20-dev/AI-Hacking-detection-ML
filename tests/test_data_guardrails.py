"""Tests for training data source guardrails."""
from pathlib import Path

import pytest

from src.data_guardrails import (
    assert_allowed_training_path,
    assert_allowed_training_paths,
)


def test_blocks_fp_test_dataset():
    with pytest.raises(ValueError, match="fp_test_500k"):
        assert_allowed_training_path("datasets/fp_test_500k.jsonl", context="unit-test")


def test_blocks_holdout_directory_children():
    with pytest.raises(ValueError, match="holdout_test"):
        assert_allowed_training_path(
            Path("datasets/holdout_test") / "payload_holdout.jsonl",
            context="unit-test",
        )


def test_allows_regular_training_sources():
    assert_allowed_training_paths(
        [
            "datasets/security_payloads/injection/sqli.txt",
            "datasets/live_benign/common_crawl_urls.jsonl",
        ],
        context="unit-test",
    )
