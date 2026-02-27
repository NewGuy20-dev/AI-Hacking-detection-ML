#!/usr/bin/env python3
"""Publish the AI hacking detection model stack to Hugging Face without cloning."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as importlib_metadata
import io
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


HF_REPO_ID = "GRK2012/ai-hacking-detection-system"
SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
HASH_CHUNK_SIZE = 1024 * 1024
DEFAULT_THRESHOLD = 0.5
DEFAULT_OUTPUT_TYPE = "binary_probability"
DEFAULT_INPUT_SHAPE = "unknown"

THRESHOLD_PATHS: tuple[tuple[str, ...], ...] = (
    ("threshold",),
    ("decision_threshold",),
    ("score_threshold",),
    ("model", "threshold"),
    ("inference", "threshold"),
    ("thresholds", "default"),
    ("calibration", "threshold"),
)

OUTPUT_TYPE_PATHS: tuple[tuple[str, ...], ...] = (
    ("output_type",),
    ("prediction_type",),
    ("task_type",),
    ("model", "output_type"),
    ("inference", "output_type"),
)

INPUT_SHAPE_PATHS: tuple[tuple[str, ...], ...] = (
    ("input_shape",),
    ("input_size",),
    ("shape",),
    ("model", "input_shape"),
)


@dataclass(frozen=True)
class ModelSpec:
    """Source and manifest contract for one model family."""

    name: str
    framework: str
    model_file: str
    scaler_file: str | None
    config_file: str | None
    threshold_file: str
    required_files: tuple[str, ...]


@dataclass(frozen=True)
class HashRecord:
    """Hash metadata for model/scaler artifacts."""

    path_in_repo: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class ModelPublishInfo:
    """Resolved model metadata ready for manifest and upload."""

    spec: ModelSpec
    files: Mapping[str, Path]
    threshold: Any
    input_shape: Any | None
    output_type: str
    sha256_model: str
    sha256_scaler: str | None


@dataclass(frozen=True)
class UploadItem:
    """One file to add to the remote repo."""

    path_in_repo: str
    source_path: Path | None
    content: bytes | None
    size_bytes: int


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="payload_cnn",
        framework="pytorch",
        model_file="model.pt",
        scaler_file=None,
        config_file="config.json",
        threshold_file="threshold.yaml",
        required_files=("model.pt", "config.json", "threshold.yaml"),
    ),
    ModelSpec(
        name="url_cnn",
        framework="pytorch",
        model_file="model.pt",
        scaler_file=None,
        config_file="config.json",
        threshold_file="threshold.yaml",
        required_files=("model.pt", "config.json", "threshold.yaml"),
    ),
    ModelSpec(
        name="timeseries_lstm",
        framework="pytorch",
        model_file="model.pt",
        scaler_file=None,
        config_file="config.json",
        threshold_file="threshold.yaml",
        required_files=("model.pt", "config.json", "threshold.yaml"),
    ),
    ModelSpec(
        name="meta_classifier",
        framework="pytorch",
        model_file="model.pt",
        scaler_file=None,
        config_file="config.json",
        threshold_file="threshold.yaml",
        required_files=("model.pt", "config.json", "threshold.yaml"),
    ),
    ModelSpec(
        name="network_intrusion_rf",
        framework="sklearn",
        model_file="model.pkl",
        scaler_file="scaler.pkl",
        config_file=None,
        threshold_file="threshold.yaml",
        required_files=("model.pkl", "scaler.pkl", "sklearn_version.txt", "threshold.yaml"),
    ),
    ModelSpec(
        name="host_behavior_rf",
        framework="sklearn",
        model_file="model.pkl",
        scaler_file="scaler.pkl",
        config_file=None,
        threshold_file="threshold.yaml",
        required_files=("model.pkl", "scaler.pkl", "sklearn_version.txt", "threshold.yaml"),
    ),
    ModelSpec(
        name="fraud_detection_xgb",
        framework="xgboost",
        model_file="model.pkl",
        scaler_file="scaler.pkl",
        config_file=None,
        threshold_file="threshold.yaml",
        required_files=("model.pkl", "scaler.pkl", "xgboost_version.txt", "threshold.yaml"),
    ),
)

MODEL_SPEC_BY_NAME: dict[str, ModelSpec] = {spec.name: spec for spec in MODEL_SPECS}

FLAT_LAYOUT_FILE_CANDIDATES: dict[str, dict[str, tuple[str, ...]]] = {
    "payload_cnn": {"model": ("payload_cnn.pt", "payload_cnn.pth", "payload_cnn_best.pt")},
    "url_cnn": {"model": ("url_cnn.pt", "url_cnn.pth", "url_cnn_best.pt")},
    "timeseries_lstm": {"model": ("timeseries_lstm.pt", "timeseries_lstm.pth")},
    "meta_classifier": {"model": ("meta_classifier.pt", "meta_classifier.pth")},
    "network_intrusion_rf": {
        "model": ("network_intrusion_model.pkl",),
        "scaler": ("network_scaler.pkl",),
    },
    "host_behavior_rf": {
        "model": ("host_behavior_model.pkl",),
        "scaler": ("host_behavior_scaler.pkl",),
    },
    "fraud_detection_xgb": {
        "model": ("fraud_detection_model.pkl",),
        "scaler": ("fraud_scaler.pkl",),
    },
}

MODEL_THRESHOLD_KEYS: dict[str, str] = {
    "payload_cnn": "payload_cnn",
    "url_cnn": "url_cnn",
    "timeseries_lstm": "default",
    "meta_classifier": "ensemble",
    "network_intrusion_rf": "network",
    "host_behavior_rf": "default",
    "fraud_detection_xgb": "fraud",
}

HF_TOKEN_ENV_KEYS: tuple[str, ...] = (
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_TOKEN",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build and publish the AI model stack to Hugging Face."
    )
    parser.add_argument(
        "--source_dir",
        required=True,
        help=(
            "Path to local model artifacts. Supports strict mirror layout and legacy flat "
            "artifact layout."
        ),
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Semantic version MAJOR.MINOR.PATCH (example: 1.2.0).",
    )
    parser.add_argument(
        "--wipe_remote",
        action="store_true",
        help="Delete all existing remote files before publishing.",
    )
    return parser.parse_args(argv)


def validate_semver(version: str) -> None:
    """Validate semantic version format."""
    if not SEMVER_RE.fullmatch(version):
        raise ValueError(
            f"Invalid version '{version}'. Expected semantic version MAJOR.MINOR.PATCH."
        )


def sha256_stream(path: Path) -> str:
    """Compute SHA256 hash via streaming reads."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and require a top-level mapping."""
    with path.open("r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected YAML mapping in '{path}', found {type(parsed).__name__}.")
    return parsed


def load_json_mapping(path: Path) -> dict[str, Any]:
    """Load a JSON file and require a top-level mapping."""
    with path.open("r", encoding="utf-8") as handle:
        parsed = json.load(handle)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object in '{path}', found {type(parsed).__name__}.")
    return parsed


def get_nested_value(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any | None:
    """Resolve a nested value path from a mapping."""
    cursor: Any = mapping
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def get_required_value(
    mapping: Mapping[str, Any],
    candidate_paths: Iterable[tuple[str, ...]],
    *,
    context: str,
) -> Any:
    """Get a required value from any candidate nested path."""
    for path in candidate_paths:
        value = get_nested_value(mapping, path)
        if value is not None:
            return value
    joined = ", ".join(".".join(path) for path in candidate_paths)
    raise ValueError(f"Missing required value for {context}. Checked keys: {joined}.")


def get_required_string(
    mapping: Mapping[str, Any],
    candidate_paths: Iterable[tuple[str, ...]],
    *,
    context: str,
) -> str:
    """Get a required non-empty string value."""
    value = get_required_value(mapping, candidate_paths, context=context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty string for {context}, found {value!r}.")
    return value.strip()


def find_first_existing(base_dir: Path, candidates: Iterable[str]) -> Path | None:
    """Return first existing file path from a candidate name list."""
    for candidate in candidates:
        path = base_dir / candidate
        if path.is_file():
            return path
    return None


def load_threshold_defaults() -> dict[str, float]:
    """Load optional threshold defaults from project config."""
    defaults: dict[str, float] = {"default": DEFAULT_THRESHOLD}
    config_path = Path(__file__).resolve().parents[1] / "configs/inference/optimal_thresholds.json"
    if not config_path.is_file():
        return defaults

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return defaults

    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping):
        return defaults

    for key, value in thresholds.items():
        if isinstance(value, (int, float)):
            defaults[str(key)] = float(value)
    return defaults


def choose_threshold_value(model_name: str, defaults: Mapping[str, float]) -> float:
    """Select threshold for a model from defaults with fallback."""
    key = MODEL_THRESHOLD_KEYS.get(model_name, "default")
    if key in defaults:
        return defaults[key]
    if "default" in defaults:
        return defaults["default"]
    return DEFAULT_THRESHOLD


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    """Write YAML mapping to path."""
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(payload),
            handle,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
        )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON mapping to path."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2)
        handle.write("\n")


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple .env file into key/value pairs."""
    entries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            entries[key] = value
    return entries


def resolve_hf_token(repo_root: Path) -> tuple[str | None, str]:
    """
    Resolve Hugging Face token from environment, then .env.local, then environment fallback.

    Returns (token, source). If token is None, caller should rely on existing HF auth cache.
    """
    for key in HF_TOKEN_ENV_KEYS:
        token = os.getenv(key)
        if token and token.strip():
            return token.strip(), f"environment:{key}"

    env_file = repo_root / ".env.local"
    if env_file.is_file():
        env_entries = parse_env_file(env_file)
        for key in HF_TOKEN_ENV_KEYS:
            token = env_entries.get(key)
            if token and token.strip():
                normalized = token.strip()
                os.environ["HF_TOKEN"] = normalized
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = normalized
                return normalized, f"file:{env_file.name}:{key}"

    return None, "huggingface_cached_auth"


def stage_flat_layout_source(
    source_dir: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path, list[str]]:
    """
    Stage legacy flat model artifacts into strict mirror layout.

    Returns the temporary directory manager, staged root path, and warning messages.
    """
    missing: list[str] = []
    selected_paths: dict[str, dict[str, Path]] = {}

    for model_name, candidates in FLAT_LAYOUT_FILE_CANDIDATES.items():
        model_path = find_first_existing(source_dir, candidates["model"])
        if model_path is None:
            missing.append(
                f"{model_name}: missing model file candidates {', '.join(candidates['model'])}"
            )
            continue

        selected: dict[str, Path] = {"model": model_path}
        scaler_candidates = candidates.get("scaler")
        if scaler_candidates is not None:
            scaler_path = find_first_existing(source_dir, scaler_candidates)
            if scaler_path is None:
                missing.append(
                    f"{model_name}: missing scaler file candidates {', '.join(scaler_candidates)}"
                )
                continue
            selected["scaler"] = scaler_path
        selected_paths[model_name] = selected

    if missing:
        details = "\n".join(f"- {item}" for item in missing)
        raise ValueError(
            "Source directory does not match strict layout or supported flat layout:\n"
            f"{details}"
        )

    tmpdir = tempfile.TemporaryDirectory(prefix="hf_model_publish_")
    staged_root = Path(tmpdir.name)
    warnings: list[str] = []
    threshold_defaults = load_threshold_defaults()
    sklearn_version = get_package_version("scikit-learn", "sklearn")
    xgboost_version = get_package_version("xgboost", "xgboost")

    for model_name, selected in selected_paths.items():
        spec = MODEL_SPEC_BY_NAME[model_name]
        model_dir = staged_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(selected["model"], model_dir / spec.model_file)
        if spec.scaler_file is not None:
            scaler_source = selected.get("scaler")
            if scaler_source is None:
                raise ValueError(f"Missing scaler selection for {model_name}.")
            shutil.copy2(scaler_source, model_dir / spec.scaler_file)

        threshold_path = model_dir / spec.threshold_file
        threshold_payload = {
            "threshold": choose_threshold_value(model_name, threshold_defaults),
            "output_type": DEFAULT_OUTPUT_TYPE,
            "source": "auto_generated",
        }
        write_yaml(threshold_path, threshold_payload)
        warnings.append(f"{model_name}: generated {spec.threshold_file}")

        if spec.config_file is not None:
            config_payload = {
                "model_name": model_name,
                "framework": spec.framework,
                "input_shape": DEFAULT_INPUT_SHAPE,
                "output_type": DEFAULT_OUTPUT_TYPE,
                "source": "auto_generated",
            }
            write_json(model_dir / spec.config_file, config_payload)
            warnings.append(f"{model_name}: generated {spec.config_file}")

        if spec.framework == "sklearn":
            (model_dir / "sklearn_version.txt").write_text(f"{sklearn_version}\n", encoding="utf-8")
            warnings.append(f"{model_name}: generated sklearn_version.txt")
        if spec.framework == "xgboost":
            (model_dir / "xgboost_version.txt").write_text(f"{xgboost_version}\n", encoding="utf-8")
            warnings.append(f"{model_name}: generated xgboost_version.txt")

    return tmpdir, staged_root, warnings


def resolve_and_validate_source(source_dir: Path) -> tuple[list[ModelPublishInfo], list[HashRecord]]:
    """Validate local source layout and extract model metadata."""
    if not source_dir.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise ValueError(f"Source path is not a directory: {source_dir}")

    missing_files: list[str] = []
    path_usage: dict[Path, list[str]] = {}
    model_infos: list[ModelPublishInfo] = []
    hash_records: list[HashRecord] = []

    for spec in MODEL_SPECS:
        model_dir = source_dir / spec.name
        if not model_dir.is_dir():
            missing_files.append(f"{spec.name}/ (directory missing)")
            continue

        file_map: dict[str, Path] = {}
        for filename in spec.required_files:
            artifact_path = model_dir / filename
            path_in_repo = f"{spec.name}/{filename}"
            if not artifact_path.is_file():
                missing_files.append(path_in_repo)
                continue
            file_map[filename] = artifact_path
            resolved = artifact_path.resolve(strict=True)
            path_usage.setdefault(resolved, []).append(path_in_repo)

        if len(file_map) != len(spec.required_files):
            continue

        threshold_doc = load_yaml_mapping(file_map[spec.threshold_file])
        threshold = get_required_value(
            threshold_doc,
            THRESHOLD_PATHS,
            context=f"{spec.name} threshold ({file_map[spec.threshold_file]})",
        )

        if spec.config_file is not None:
            config_doc = load_json_mapping(file_map[spec.config_file])
            input_shape = get_required_value(
                config_doc,
                INPUT_SHAPE_PATHS,
                context=f"{spec.name} input_shape ({file_map[spec.config_file]})",
            )

            try:
                output_type = get_required_string(
                    config_doc,
                    OUTPUT_TYPE_PATHS,
                    context=f"{spec.name} output_type ({file_map[spec.config_file]})",
                )
            except ValueError:
                output_type = get_required_string(
                    threshold_doc,
                    OUTPUT_TYPE_PATHS,
                    context=f"{spec.name} output_type ({file_map[spec.threshold_file]})",
                )
        else:
            input_shape = None
            output_type = get_required_string(
                threshold_doc,
                OUTPUT_TYPE_PATHS,
                context=f"{spec.name} output_type ({file_map[spec.threshold_file]})",
            )

        model_path = file_map[spec.model_file]
        model_hash = sha256_stream(model_path)
        hash_records.append(
            HashRecord(
                path_in_repo=f"{spec.name}/{spec.model_file}",
                sha256=model_hash,
                size_bytes=model_path.stat().st_size,
            )
        )

        scaler_hash: str | None = None
        if spec.scaler_file is not None:
            scaler_path = file_map[spec.scaler_file]
            scaler_hash = sha256_stream(scaler_path)
            hash_records.append(
                HashRecord(
                    path_in_repo=f"{spec.name}/{spec.scaler_file}",
                    sha256=scaler_hash,
                    size_bytes=scaler_path.stat().st_size,
                )
            )

        model_infos.append(
            ModelPublishInfo(
                spec=spec,
                files=file_map,
                threshold=threshold,
                input_shape=input_shape,
                output_type=output_type,
                sha256_model=model_hash,
                sha256_scaler=scaler_hash,
            )
        )

    if missing_files:
        details = "\n".join(f"- {item}" for item in missing_files)
        raise ValueError(f"Missing required model artifacts in source_dir:\n{details}")

    duplicate_sources = {
        source: targets for source, targets in path_usage.items() if len(targets) > 1
    }
    if duplicate_sources:
        lines = [
            f"- {source} referenced by: {', '.join(paths)}"
            for source, paths in duplicate_sources.items()
        ]
        raise ValueError(
            "Duplicate source files detected (same physical file used by multiple targets):\n"
            + "\n".join(lines)
        )

    if len(model_infos) != len(MODEL_SPECS):
        resolved_names = {info.spec.name for info in model_infos}
        missing_specs = [spec.name for spec in MODEL_SPECS if spec.name not in resolved_names]
        raise ValueError(f"Failed to resolve model specs: {', '.join(missing_specs)}")

    return model_infos, hash_records


def get_package_version(distribution: str, import_name: str | None = None) -> str:
    """Return installed package version or 'not_installed'."""
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        if import_name is None:
            return "not_installed"

    if import_name is None:
        return "not_installed"

    try:
        module = importlib.import_module(import_name)
    except Exception:
        return "not_installed"

    module_version = getattr(module, "__version__", None)
    if isinstance(module_version, str) and module_version.strip():
        return module_version.strip()
    return "not_installed"


def build_manifest(version: str, model_infos: list[ModelPublishInfo]) -> dict[str, Any]:
    """Build manifest payload for model_manifest.yaml."""
    manifest: dict[str, Any] = {
        "version": version,
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": get_package_version("torch", "torch"),
            "sklearn_version": get_package_version("scikit-learn", "sklearn"),
            "xgboost_version": get_package_version("xgboost", "xgboost"),
        },
        "models": {},
    }

    model_entries: dict[str, Any] = {}
    for info in model_infos:
        if info.spec.framework == "pytorch":
            model_entries[info.spec.name] = {
                "framework": "pytorch",
                "sha256": info.sha256_model,
                "threshold": info.threshold,
                "input_shape": info.input_shape,
                "output_type": info.output_type,
            }
            continue

        model_entry: dict[str, Any] = {
            "framework": info.spec.framework,
            "sha256_model": info.sha256_model,
            "threshold": info.threshold,
            "output_type": info.output_type,
        }
        if info.sha256_scaler is None:
            raise ValueError(f"Missing scaler hash for model '{info.spec.name}'.")
        model_entry["sha256_scaler"] = info.sha256_scaler
        model_entries[info.spec.name] = model_entry

    manifest["models"] = model_entries
    return manifest


def render_readme(version: str) -> str:
    """Create repository README content."""
    return f"""---
tags:
- cybersecurity
- intrusion-detection
- anomaly-detection
- machine-learning
- pytorch
- scikit-learn
- xgboost
---

# AI Hacking Detection System

This repository hosts the published production model stack for the AI hacking detection system.
It includes deep learning and classical ML artifacts for payload, URL, timeseries, meta, network,
host behavior, and fraud detection inference pipelines.

## Version

`{version}`

## Integrity Hashing

`model_manifest.yaml` records SHA256 hashes for every model file and scaler file.
Use the manifest as the source of truth to verify artifact integrity after download.

## Hash Validation

Example shell verification:

```bash
sha256sum payload_cnn/model.pt network_intrusion_rf/model.pkl network_intrusion_rf/scaler.pkl
```

Example Python verification:

```python
import hashlib
from pathlib import Path

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print(sha256("payload_cnn/model.pt"))
```
"""


def build_generated_files(version: str, manifest: Mapping[str, Any]) -> dict[str, bytes]:
    """Create bytes content for generated repository files."""
    manifest_bytes = yaml.safe_dump(
        manifest,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    ).encode("utf-8")
    version_bytes = f"{version}\n".encode("utf-8")
    readme_bytes = render_readme(version).encode("utf-8")
    return {
        "model_manifest.yaml": manifest_bytes,
        "VERSION": version_bytes,
        "README.md": readme_bytes,
    }


def build_upload_items(
    model_infos: list[ModelPublishInfo],
    generated_files: Mapping[str, bytes],
) -> list[UploadItem]:
    """Create upload list with duplicate path protection."""
    items: list[UploadItem] = []
    used_paths: set[str] = set()

    for info in model_infos:
        for filename in info.spec.required_files:
            source_path = info.files[filename]
            path_in_repo = f"{info.spec.name}/{filename}"
            if path_in_repo in used_paths:
                raise ValueError(f"Duplicate upload path detected: {path_in_repo}")
            used_paths.add(path_in_repo)
            items.append(
                UploadItem(
                    path_in_repo=path_in_repo,
                    source_path=source_path,
                    content=None,
                    size_bytes=source_path.stat().st_size,
                )
            )

    for path_in_repo, content in generated_files.items():
        if path_in_repo in used_paths:
            raise ValueError(f"Duplicate upload path detected: {path_in_repo}")
        used_paths.add(path_in_repo)
        items.append(
            UploadItem(
                path_in_repo=path_in_repo,
                source_path=None,
                content=content,
                size_bytes=len(content),
            )
        )

    return items


def human_size(size_bytes: int) -> str:
    """Render bytes in a readable size format."""
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


def print_upload_summary(
    *,
    repo_id: str,
    version: str,
    wipe_remote: bool,
    upload_items: list[UploadItem],
    hash_records: list[HashRecord],
) -> None:
    """Print a concise summary before upload."""
    total_size = sum(item.size_bytes for item in upload_items)
    print(f"Repository: {repo_id}", flush=True)
    print(f"Version: {version}", flush=True)
    print(f"Wipe remote: {'yes' if wipe_remote else 'no'}", flush=True)
    print(f"Files to upload: {len(upload_items)} ({human_size(total_size)})", flush=True)
    print("", flush=True)
    print("Computed SHA256 hashes:", flush=True)
    for record in hash_records:
        print(
            f"  {record.path_in_repo}: {record.sha256} ({human_size(record.size_bytes)})",
            flush=True,
        )
    print("", flush=True)


def publish_to_hf(
    *,
    repo_id: str,
    version: str,
    wipe_remote: bool,
    upload_items: list[UploadItem],
    hf_token: str | None,
) -> str:
    """Publish files to Hugging Face Hub in one atomic commit."""
    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
        from huggingface_hub.utils import HfHubHTTPError
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install it with: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=hf_token)
    operations: list[Any] = []
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        if wipe_remote:
            remote_paths = api.list_repo_files(repo_id=repo_id, repo_type="model")
            operations.extend(
                CommitOperationDelete(path_in_repo=path_in_repo)
                for path_in_repo in remote_paths
            )

        for item in upload_items:
            if item.source_path is not None:
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=item.path_in_repo,
                        path_or_fileobj=str(item.source_path),
                    )
                )
            elif item.content is not None:
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=item.path_in_repo,
                        path_or_fileobj=io.BytesIO(item.content),
                    )
                )
            else:
                raise RuntimeError(f"Upload item has no content source: {item.path_in_repo}")

        commit_info = api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            operations=operations,
            commit_message=f"Publish model stack v{version} with hashing and manifest",
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(f"Hugging Face API error: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to publish to Hugging Face: {exc}") from exc

    commit_url = getattr(commit_info, "commit_url", "")
    commit_oid = getattr(commit_info, "oid", "")
    if isinstance(commit_url, str) and commit_url:
        return commit_url
    if isinstance(commit_oid, str) and commit_oid:
        return commit_oid
    return "commit created"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    staging_tmpdir: tempfile.TemporaryDirectory[str] | None = None
    staging_warnings: list[str] = []
    try:
        repo_root = Path(__file__).resolve().parents[1]
        source_dir = Path(args.source_dir).expanduser().resolve()
        validate_semver(args.version)

        try:
            model_infos, hash_records = resolve_and_validate_source(source_dir)
        except ValueError as strict_error:
            try:
                staging_tmpdir, staged_root, staging_warnings = stage_flat_layout_source(source_dir)
                model_infos, hash_records = resolve_and_validate_source(staged_root)
            except ValueError:
                raise strict_error

        manifest = build_manifest(args.version, model_infos)
        generated_files = build_generated_files(args.version, manifest)
        upload_items = build_upload_items(model_infos, generated_files)

        if staging_warnings:
            print(
                "Source layout mode: flat (auto-staged to required repo structure)",
                flush=True,
            )
            for warning in staging_warnings:
                print(f"  note: {warning}", flush=True)
            print("", flush=True)

        print_upload_summary(
            repo_id=HF_REPO_ID,
            version=args.version,
            wipe_remote=args.wipe_remote,
            upload_items=upload_items,
            hash_records=hash_records,
        )

        hf_token, hf_auth_source = resolve_hf_token(repo_root)
        print(f"HF auth source: {hf_auth_source}", flush=True)
        commit_ref = publish_to_hf(
            repo_id=HF_REPO_ID,
            version=args.version,
            wipe_remote=args.wipe_remote,
            upload_items=upload_items,
            hf_token=hf_token,
        )
        print(f"Commit success: {commit_ref}", flush=True)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if staging_tmpdir is not None:
            staging_tmpdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
