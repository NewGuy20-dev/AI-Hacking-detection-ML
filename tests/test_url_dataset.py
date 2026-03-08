"""Regression coverage for the real URL streaming dataset."""
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from src.data.url_dataset import RealURLDataset


def _write_jsonl(path: Path) -> None:
    path.write_text(
        '{"url": "http://example.com"}\n'
        'not-json\n'
        '{"text": "https://example.org/path"}\n',
        encoding="utf-8",
    )


def test_read_jsonl_close_does_not_ignore_generator_exit(temp_dir):
    path = temp_dir / "urls.jsonl"
    _write_jsonl(path)
    dataset = RealURLDataset([path], [path], samples_per_epoch=2)

    stream = dataset._read_jsonl(path)

    assert next(stream) == "http://example.com"
    stream.close()


def test_read_jsonl_skips_malformed_rows(temp_dir):
    path = temp_dir / "urls.jsonl"
    _write_jsonl(path)
    dataset = RealURLDataset([path], [path], samples_per_epoch=2)

    assert list(dataset._read_jsonl(path)) == [
        "http://example.com",
        "https://example.org/path",
    ]


def test_stream_close_propagates_clean_shutdown(temp_dir):
    path = temp_dir / "urls.jsonl"
    _write_jsonl(path)
    dataset = RealURLDataset([path], [path], samples_per_epoch=2)

    stream = dataset._stream([path])

    assert next(stream) == "http://example.com"
    stream.close()


def test_dataset_yields_token_and_label_pairs(temp_dir):
    malicious_path = temp_dir / "malicious.jsonl"
    benign_path = temp_dir / "benign.jsonl"
    malicious_path.write_text('{"url": "http://bad.example"}\n', encoding="utf-8")
    benign_path.write_text('{"url": "http://good.example"}\n', encoding="utf-8")
    dataset = RealURLDataset([malicious_path], [benign_path], samples_per_epoch=2)

    samples = list(iter(dataset))

    assert len(samples) == 2
    first_tokens, first_label = samples[0]
    second_tokens, second_label = samples[1]

    assert isinstance(first_tokens, torch.Tensor)
    assert isinstance(second_tokens, torch.Tensor)
    assert first_tokens.shape[0] == dataset.max_len
    assert second_tokens.shape[0] == dataset.max_len
    assert {first_label.item(), second_label.item()} == {0.0, 1.0}
