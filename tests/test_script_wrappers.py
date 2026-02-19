"""Smoke checks for compatibility wrapper scripts."""
from pathlib import Path
import re


WRAPPERS = [
    "scripts/stress_test_v14.py",
    "scripts/stress_test.py",
    "scripts/train_rtx3050.py",
    "scripts/evaluate_models.py",
    "scripts/download_url_datasets.py",
    "scripts/download_missing_datasets.py",
    "scripts/generate_benign_data.py",
    "scripts/generate_adversarial_benign.py",
    "scripts/generate_500k_benign_test.py",
    "scripts/generate_60m_benign.py",
    "scripts/generate_malicious_urls.py",
    "scripts/validate_models.py",
    "scripts/prepare_url_data.py",
]


def test_wrapper_targets_exist():
    """Each wrapper should point to an existing canonical script path."""
    root = Path(__file__).resolve().parents[1]
    for wrapper in WRAPPERS:
        wrapper_path = root / wrapper
        assert wrapper_path.exists(), f"Missing wrapper: {wrapper}"
        text = wrapper_path.read_text(encoding='utf-8')
        match = re.search(r'TARGET = ROOT / "([^"]+)"', text)
        assert match, f"Wrapper missing TARGET assignment: {wrapper}"
        target = root / match.group(1)
        assert target.exists(), f"Wrapper target missing for {wrapper}: {target}"
