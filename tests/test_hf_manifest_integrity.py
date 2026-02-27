"""Validate Hugging Face model artifact integrity against manifest hashes."""

from tests.hf_validation.manifest_validator import validate_manifest_hashes


def test_hf_manifest_hashes_match_downloaded_artifacts(hf_artifacts) -> None:
    """Fail when downloaded model/scaler hashes diverge from model_manifest.yaml."""
    mismatches = validate_manifest_hashes(hf_artifacts)
    assert not mismatches, (
        "Manifest SHA256 mismatch detected:\n"
        + "\n".join(
            f"- {item.model} [{item.repo_path}] expected={item.expected} actual={item.actual}"
            for item in mismatches
        )
    )

