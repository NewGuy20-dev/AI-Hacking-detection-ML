"""Smoke evaluation helpers for Hugging Face model artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from src.torch_models.meta_classifier import MetaClassifier
from src.torch_models.payload_cnn import PayloadCNN
from src.torch_models.timeseries_lstm import TimeSeriesLSTM
from src.torch_models.url_cnn import URLCNN
from tests.hf_validation.hf_loader import TORCH_MODELS, DownloadedArtifacts

THRESHOLD_KEYS: tuple[tuple[str, ...], ...] = (
    ("threshold",),
    ("decision_threshold",),
    ("score_threshold",),
    ("thresholds", "default"),
)


@dataclass(frozen=True)
class SmokeResult:
    """Result of a model smoke test."""

    model_name: str
    raw_output_shape: tuple[int, ...]
    probability_shape: tuple[int, ...]
    min_probability: float
    max_probability: float


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} is not a JSON object.")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def _nested_get(mapping: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    cursor: Any = mapping
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str):
        try:
            parsed = int(value)
            if parsed > 0:
                return parsed
        except ValueError:
            return default
    return default


def _resolve_threshold(threshold_path: Path, default: float = 0.5) -> float:
    payload = _load_yaml(threshold_path)
    for key_path in THRESHOLD_KEYS:
        value = _nested_get(payload, key_path)
        if isinstance(value, (int, float)):
            return float(value)
    return default


def _extract_state_dict(load_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(load_obj, torch.nn.Module):
        return dict(load_obj.state_dict())

    if isinstance(load_obj, dict):
        for key in ("state_dict", "model_state_dict"):
            candidate = load_obj.get(key)
            if isinstance(candidate, dict):
                return dict(candidate)
        if load_obj and all(isinstance(v, torch.Tensor) for v in load_obj.values()):
            return dict(load_obj)

    raise ValueError("Unsupported torch artifact format. Expected state_dict-compatible checkpoint.")


def _load_torch_weights(model: torch.nn.Module, model_path: Path) -> None:
    loaded = torch.load(model_path, map_location="cpu")
    state_dict = _extract_state_dict(loaded)
    if not state_dict:
        raise ValueError(f"Empty state_dict loaded from {model_path}")

    result = model.load_state_dict(state_dict, strict=False)
    matched_any = len(result.missing_keys) < len(model.state_dict())
    if not matched_any:
        raise ValueError(
            f"Loaded state_dict from {model_path} did not match architecture keys "
            f"(unexpected={len(result.unexpected_keys)}, missing={len(result.missing_keys)})."
        )


def _instantiate_model(model_name: str, config: dict[str, Any]) -> torch.nn.Module:
    if model_name == "payload_cnn":
        return PayloadCNN(
            vocab_size=_as_int(config.get("vocab_size"), 256),
            embed_dim=_as_int(config.get("embed_dim"), 128),
            num_filters=_as_int(config.get("num_filters"), 256),
            max_len=_as_int(config.get("max_len"), 500),
        )
    if model_name == "url_cnn":
        return URLCNN(
            vocab_size=_as_int(config.get("vocab_size"), 128),
            embed_dim=_as_int(config.get("embed_dim"), 64),
            max_len=_as_int(config.get("max_len"), 200),
        )
    if model_name == "timeseries_lstm":
        return TimeSeriesLSTM(
            input_dim=_as_int(config.get("input_dim"), 8),
            hidden_dim=_as_int(config.get("hidden_dim"), 64),
            num_layers=_as_int(config.get("num_layers"), 2),
            dropout=float(config.get("dropout", 0.4)),
        )
    if model_name == "meta_classifier":
        return MetaClassifier(
            num_models=_as_int(config.get("num_models"), 5),
        )
    raise ValueError(f"Unsupported smoke test model: {model_name}")


def _synthetic_input(model_name: str, config: dict[str, Any]) -> torch.Tensor:
    generator = torch.Generator().manual_seed(1337)
    if model_name == "payload_cnn":
        seq_len = _as_int(config.get("max_len"), 500)
        vocab_size = _as_int(config.get("vocab_size"), 256)
        return torch.randint(1, vocab_size, (1, seq_len), generator=generator, dtype=torch.long)
    if model_name == "url_cnn":
        seq_len = _as_int(config.get("max_len"), 200)
        vocab_size = _as_int(config.get("vocab_size"), 128)
        return torch.randint(1, vocab_size, (1, seq_len), generator=generator, dtype=torch.long)
    if model_name == "timeseries_lstm":
        seq_len = _as_int(config.get("seq_len"), 60)
        input_dim = _as_int(config.get("input_dim"), 8)
        return torch.rand((1, seq_len, input_dim), generator=generator, dtype=torch.float32)
    if model_name == "meta_classifier":
        num_models = _as_int(config.get("num_models"), 5)
        values = torch.linspace(0.1, 0.9, steps=num_models, dtype=torch.float32)
        return values.unsqueeze(0)
    raise ValueError(f"Unsupported model for synthetic input: {model_name}")


def run_smoke_validation(artifacts: DownloadedArtifacts) -> dict[str, SmokeResult]:
    """Run one deterministic smoke forward pass per torch model."""
    results: dict[str, SmokeResult] = {}
    for model_name in TORCH_MODELS:
        config = _load_json(artifacts.path_for(f"{model_name}/config.json"))
        _resolve_threshold(artifacts.path_for(f"{model_name}/threshold.yaml"))

        model = _instantiate_model(model_name, config)
        _load_torch_weights(model, artifacts.path_for(f"{model_name}/model.pt"))
        model.eval()

        sample = _synthetic_input(model_name, config)
        with torch.inference_mode():
            logits = model(sample)
            probs = torch.sigmoid(logits.reshape(1, -1))

        if probs.numel() != 1:
            raise AssertionError(
                f"{model_name} expected exactly 1 output value, got shape {tuple(probs.shape)}"
            )
        if probs.min().item() < 0.0 or probs.max().item() > 1.0:
            raise AssertionError(
                f"{model_name} probability out of range [0, 1]: "
                f"min={probs.min().item():.6f}, max={probs.max().item():.6f}"
            )
        if model_name == "meta_classifier" and tuple(probs.shape) != (1, 1):
            raise AssertionError(
                f"meta_classifier output must be shape (1,1), got {tuple(probs.shape)}"
            )

        results[model_name] = SmokeResult(
            model_name=model_name,
            raw_output_shape=tuple(logits.shape),
            probability_shape=tuple(probs.shape),
            min_probability=probs.min().item(),
            max_probability=probs.max().item(),
        )
    return results


def load_torch_models_for_eval(artifacts: DownloadedArtifacts) -> dict[str, tuple[torch.nn.Module, float]]:
    """Load all torch models and their thresholds for benchmark evaluation."""
    loaded: dict[str, tuple[torch.nn.Module, float]] = {}
    for model_name in TORCH_MODELS:
        config = _load_json(artifacts.path_for(f"{model_name}/config.json"))
        model = _instantiate_model(model_name, config)
        _load_torch_weights(model, artifacts.path_for(f"{model_name}/model.pt"))
        model.eval()
        threshold = _resolve_threshold(artifacts.path_for(f"{model_name}/threshold.yaml"))
        loaded[model_name] = (model, threshold)
    return loaded
