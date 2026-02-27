"""Download helpers for Hugging Face model repository validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

DEFAULT_REPO_ID = "GRK2012/ai-hacking-detection-system"
HF_REPO_ID_ENV = "HF_REPO_ID"
HF_TOKEN_ENV = "HF_TOKEN"

TORCH_MODELS: tuple[str, ...] = (
    "payload_cnn",
    "url_cnn",
    "timeseries_lstm",
    "meta_classifier",
)

CLASSICAL_MODELS: tuple[str, ...] = (
    "network_intrusion_rf",
    "host_behavior_rf",
    "fraud_detection_xgb",
)

REQUIRED_REPO_FILES: tuple[str, ...] = (
    "payload_cnn/model.pt",
    "payload_cnn/config.json",
    "payload_cnn/threshold.yaml",
    "url_cnn/model.pt",
    "url_cnn/config.json",
    "url_cnn/threshold.yaml",
    "timeseries_lstm/model.pt",
    "timeseries_lstm/config.json",
    "timeseries_lstm/threshold.yaml",
    "meta_classifier/model.pt",
    "meta_classifier/config.json",
    "meta_classifier/threshold.yaml",
    "network_intrusion_rf/model.pkl",
    "network_intrusion_rf/scaler.pkl",
    "network_intrusion_rf/sklearn_version.txt",
    "network_intrusion_rf/threshold.yaml",
    "host_behavior_rf/model.pkl",
    "host_behavior_rf/scaler.pkl",
    "host_behavior_rf/sklearn_version.txt",
    "host_behavior_rf/threshold.yaml",
    "fraud_detection_xgb/model.pkl",
    "fraud_detection_xgb/scaler.pkl",
    "fraud_detection_xgb/xgboost_version.txt",
    "fraud_detection_xgb/threshold.yaml",
    "model_manifest.yaml",
    "VERSION",
    "README.md",
)


def normalize_token(raw_token: str) -> str:
    """Normalize token text by trimming whitespace and optional wrapping quotes."""
    token = raw_token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        token = token[1:-1].strip()
    return token


@dataclass(frozen=True)
class DownloadedArtifacts:
    """Locally downloaded Hugging Face repo artifacts."""

    repo_id: str
    root_dir: Path
    files: dict[str, Path]

    def path_for(self, repo_path: str) -> Path:
        """Return local path for a required repo file."""
        try:
            return self.files[repo_path]
        except KeyError as exc:
            raise KeyError(f"Downloaded artifact not found for {repo_path}") from exc


def resolve_repo_id() -> str:
    """Resolve HF repo id from environment or default."""
    repo_id = os.getenv(HF_REPO_ID_ENV, DEFAULT_REPO_ID).strip()
    if not repo_id:
        raise ValueError("HF repo id is empty. Set HF_REPO_ID or use default.")
    return repo_id


def get_hf_token() -> str:
    """Get HF token from environment."""
    token = normalize_token(os.getenv(HF_TOKEN_ENV, ""))
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Configure GitHub secret HF_TOKEN for Hugging Face access."
        )
    return token


def download_required_artifacts(
    *,
    local_dir: Path,
    repo_id: str | None = None,
    token: str | None = None,
    required_files: Iterable[str] = REQUIRED_REPO_FILES,
) -> DownloadedArtifacts:
    """Download all required files from the Hugging Face model repository."""
    resolved_repo_id = repo_id or resolve_repo_id()
    resolved_token = normalize_token(token) if token is not None else get_hf_token()
    local_dir.mkdir(parents=True, exist_ok=True)

    downloaded: dict[str, Path] = {}
    for repo_file in required_files:
        try:
            local_path = hf_hub_download(
                repo_id=resolved_repo_id,
                filename=repo_file,
                repo_type="model",
                token=resolved_token,
                local_dir=str(local_dir),
            )
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Failed to download '{repo_file}' from '{resolved_repo_id}': {exc}"
            ) from exc
        downloaded[repo_file] = Path(local_path)

    return DownloadedArtifacts(
        repo_id=resolved_repo_id,
        root_dir=local_dir,
        files=downloaded,
    )
