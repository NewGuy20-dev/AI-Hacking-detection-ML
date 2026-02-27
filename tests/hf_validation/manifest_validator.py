"""Manifest-based artifact integrity validation helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tests.hf_validation.hf_loader import CLASSICAL_MODELS, TORCH_MODELS, DownloadedArtifacts

HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class HashMismatch:
    """One manifest hash mismatch."""

    model: str
    repo_path: str
    expected: str
    actual: str


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash for a local file using streaming reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest at {path} is not a valid YAML mapping.")
    return manifest


def _expect_manifest_hash(
    manifest: dict[str, Any],
    model_name: str,
    key: str,
) -> str:
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("Manifest missing top-level 'models' mapping.")
    model_entry = models.get(model_name)
    if not isinstance(model_entry, dict):
        raise ValueError(f"Manifest missing entry for model '{model_name}'.")
    hash_value = model_entry.get(key)
    if not isinstance(hash_value, str) or not hash_value.strip():
        raise ValueError(f"Manifest missing hash key '{key}' for model '{model_name}'.")
    return hash_value.strip()


def validate_manifest_hashes(artifacts: DownloadedArtifacts) -> list[HashMismatch]:
    """Validate downloaded model/scaler hashes against model_manifest.yaml."""
    manifest_path = artifacts.path_for("model_manifest.yaml")
    manifest = _load_manifest(manifest_path)
    mismatches: list[HashMismatch] = []

    for model_name in TORCH_MODELS:
        repo_path = f"{model_name}/model.pt"
        expected = _expect_manifest_hash(manifest, model_name, "sha256")
        actual = sha256_file(artifacts.path_for(repo_path))
        if expected != actual:
            mismatches.append(
                HashMismatch(
                    model=model_name,
                    repo_path=repo_path,
                    expected=expected,
                    actual=actual,
                )
            )

    for model_name in CLASSICAL_MODELS:
        model_repo_path = f"{model_name}/model.pkl"
        scaler_repo_path = f"{model_name}/scaler.pkl"

        expected_model = _expect_manifest_hash(manifest, model_name, "sha256_model")
        actual_model = sha256_file(artifacts.path_for(model_repo_path))
        if expected_model != actual_model:
            mismatches.append(
                HashMismatch(
                    model=model_name,
                    repo_path=model_repo_path,
                    expected=expected_model,
                    actual=actual_model,
                )
            )

        expected_scaler = _expect_manifest_hash(manifest, model_name, "sha256_scaler")
        actual_scaler = sha256_file(artifacts.path_for(scaler_repo_path))
        if expected_scaler != actual_scaler:
            mismatches.append(
                HashMismatch(
                    model=model_name,
                    repo_path=scaler_repo_path,
                    expected=expected_scaler,
                    actual=actual_scaler,
                )
            )

    return mismatches

