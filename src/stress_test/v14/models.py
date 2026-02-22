"""Unified model wrapper for all V1.4 stress test models."""
import time
from pathlib import Path
from typing import Any, Tuple, Dict
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np
import json

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    joblib = None

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None
    nn = None


class ModelWrapper:
    """Unified interface for all stress-test models."""

    PYTORCH_MODELS = ['payload', 'url', 'timeseries', 'meta']
    SKLEARN_MODELS = ['fraud', 'host', 'network', 'anomaly']

    _threshold_cache = None

    def __init__(self, model_name: str, models_dir: Path = None):
        if model_name not in self.PYTORCH_MODELS + self.SKLEARN_MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        self.model_name = model_name
        self.models_dir = models_dir or Path(__file__).parent.parent.parent.parent / 'models'
        self.model = None
        self.scaler = None
        self.timeseries_norm = None
        self.device = 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu'

    @staticmethod
    def _load_thresholds() -> Dict[str, float]:
        """Load per-model decision thresholds from config/model_thresholds.json."""
        if ModelWrapper._threshold_cache is None:
            thresholds: Dict[str, float] = {}
            config_path = Path(__file__).parent.parent.parent.parent / 'config' / 'model_thresholds.json'
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f) or {}
                    if isinstance(data, dict):
                        thresholds = {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
                except Exception:
                    thresholds = {}
            ModelWrapper._threshold_cache = thresholds
        return ModelWrapper._threshold_cache

    def _get_threshold(self) -> float:
        thresholds = self._load_thresholds()
        return float(thresholds.get(self.model_name, 0.5))

    @classmethod
    def get_threshold(cls, name: str) -> float:
        thresholds = cls._load_thresholds()
        return float(thresholds.get(name, 0.5))

    def load(self) -> 'ModelWrapper':
        """Load model from disk."""
        if self.model_name in self.PYTORCH_MODELS:
            self._load_pytorch()
        else:
            self._load_sklearn()
        return self

    def _load_pytorch(self):
        """Load PyTorch model."""
        if torch is None:
            raise ModuleNotFoundError(
                "PyTorch is required for payload/url/timeseries/meta stress tests. "
                "Install dependencies from requirements.txt."
            )

        if self.model_name == 'payload':
            model_path = self.models_dir / 'payload_cnn.pt'
        elif self.model_name == 'url':
            model_path = self.models_dir / 'url_cnn.pt'
        elif self.model_name == 'timeseries':
            model_path = self.models_dir / 'timeseries_lstm.pt'
        else:  # meta
            model_path = self.models_dir / 'meta_classifier.pt'

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

        if self.model_name == 'timeseries':
            norm_path = self.models_dir / 'timeseries_norm_v1.npz'
            if not norm_path.exists():
                raise FileNotFoundError(
                    f"Timeseries normalization file not found: {norm_path}. "
                    "Re-train timeseries model to generate it."
                )
            data = np.load(norm_path)
            self.timeseries_norm = {
                'mins': data['mins'].astype(np.float32),
                'maxs': data['maxs'].astype(np.float32),
            }

    def _load_sklearn(self):
        """Load sklearn model and optional scaler."""
        if joblib is None:
            raise ModuleNotFoundError(
                "joblib is required for fraud/host/network/anomaly stress tests. "
                "Install dependencies from requirements.txt."
            )

        scaler_path = None
        if self.model_name == 'fraud':
            model_path = self.models_dir / 'fraud_detection_model.pkl'
            scaler_path = self.models_dir / 'fraud_scaler.pkl'
        elif self.model_name == 'host':
            model_path = self.models_dir / 'host_behavior_model.pkl'
            scaler_path = self.models_dir / 'host_behavior_scaler.pkl'
        elif self.model_name == 'network':
            model_path = self.models_dir / 'network_intrusion_model.pkl'
            scaler_path = self.models_dir / 'network_scaler.pkl'
        else:  # anomaly
            model_path = self.models_dir / 'anomaly_detector.pkl'

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if scaler_path and not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        from joblib import parallel_backend
        with parallel_backend('loky', n_jobs=1):
            artifact = joblib.load(model_path)
            if self.model_name == 'anomaly':
                if isinstance(artifact, dict) and 'model' in artifact:
                    self.model = artifact['model']
                    self.scaler = artifact.get('scaler')
                else:
                    self.model = artifact
                    self.scaler = getattr(artifact, 'scaler', None)
            else:
                self.model = artifact
                self.scaler = joblib.load(scaler_path)

    @staticmethod
    def _normalize_url_text(url: str) -> str:
        """Normalize URL to ASCII-safe form while preserving IDN semantics."""
        text = str(url).strip()
        try:
            parts = urlsplit(text)
            netloc = parts.netloc.encode('idna').decode('ascii') if parts.netloc else parts.netloc
            path = quote(parts.path, safe="/:@-._~!$&'()*+,;=%")
            query = quote(parts.query, safe="=&?/:@-._~!$'()*+,;%")
            fragment = quote(parts.fragment, safe=":@-._~!$&'()*+,;=%")
            return urlunsplit((parts.scheme, netloc, path, query, fragment))
        except Exception:
            return text

    def preprocess(self, input_data: Any) -> Any:
        """Convert scenario input to model-ready format."""
        if self.model_name == 'payload':
            if isinstance(input_data, str):
                if torch is None:
                    raise ModuleNotFoundError("PyTorch is required for payload preprocessing")
                indices = [ord(c) % 256 for c in input_data[:500]]
                indices += [0] * (500 - len(indices))
                return torch.tensor([indices], dtype=torch.long, device=self.device)

        elif self.model_name == 'url':
            if isinstance(input_data, str):
                if torch is None:
                    raise ModuleNotFoundError("PyTorch is required for url preprocessing")
                normalized = self._normalize_url_text(input_data)
                indices = [ord(c) % 128 for c in normalized[:200]]
                indices += [0] * (200 - len(indices))
                return torch.tensor([indices], dtype=torch.long, device=self.device)

        elif self.model_name == 'timeseries':
            if isinstance(input_data, np.ndarray):
                if torch is None:
                    raise ModuleNotFoundError("PyTorch is required for timeseries preprocessing")
                if input_data.shape == (60, 8):
                    input_data = input_data[np.newaxis, :]
                if self.timeseries_norm:
                    mins = self.timeseries_norm['mins']
                    maxs = self.timeseries_norm['maxs']
                    input_data = (input_data - mins) / (maxs - mins + 1e-8)
                    input_data = np.clip(input_data, 0.0, 1.0)
                return torch.tensor(input_data, dtype=torch.float32, device=self.device)

        elif self.model_name == 'meta':
            if isinstance(input_data, (list, np.ndarray)):
                if torch is None:
                    raise ModuleNotFoundError("PyTorch is required for meta preprocessing")
                input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
                return torch.tensor(input_data, dtype=torch.float32, device=self.device)

        elif self.model_name in ['fraud', 'host', 'network']:
            if isinstance(input_data, (list, np.ndarray)):
                input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
                return self.scaler.transform(input_data)

        elif self.model_name == 'anomaly':
            if isinstance(input_data, (list, np.ndarray)):
                input_data = np.array(input_data, dtype=np.float32).reshape(1, -1)
                if self.scaler is not None:
                    input_data = self.scaler.transform(input_data)
                return input_data

        raise ValueError(f"Invalid input format for {self.model_name}: {type(input_data)}")

    def predict(self, input_data: Any) -> Tuple[int, float, float]:
        """
        Run inference on input.

        Returns:
            (prediction, confidence, latency_ms)
        """
        start = time.perf_counter()

        try:
            processed = self.preprocess(input_data)

            if nn is not None and isinstance(self.model, nn.Module):
                with torch.no_grad():
                    logits = self.model(processed)
                    prob = torch.sigmoid(logits).item()
                threshold = self._get_threshold()
                prediction = 1 if prob > threshold else 0
            elif self.model_name == 'anomaly':
                if hasattr(self.model, 'score_samples') and hasattr(self.model, 'predict'):
                    raw_score = float(-self.model.score_samples(processed)[0])
                    pred_raw = int(self.model.predict(processed)[0])
                    prob = float(1.0 / (1.0 + np.exp(-raw_score)))
                    threshold = self._get_threshold()
                    prediction = 1 if prob > threshold else 0
                elif hasattr(self.model, 'predict'):
                    out = self.model.predict(processed)
                    if isinstance(out, dict):
                        is_anomaly = np.asarray(out.get('is_anomaly', [0])).reshape(-1)
                        score = float(np.asarray(out.get('anomaly_score', [0.0])).reshape(-1)[0])
                        prob = float(1.0 / (1.0 + np.exp(-score)))
                        threshold = self._get_threshold()
                        prediction = 1 if prob > threshold else 0
                    else:
                        pred_raw = int(np.asarray(out).reshape(-1)[0])
                        prob = float(1 if pred_raw == -1 else int(pred_raw > 0))
                        threshold = self._get_threshold()
                        prediction = 1 if prob > threshold else 0
                else:
                    raise RuntimeError("Anomaly model does not support prediction")
            else:
                prob = float(self.model.predict_proba(processed)[0, 1])
                threshold = self._get_threshold()
                prediction = 1 if prob > threshold else 0

            latency = (time.perf_counter() - start) * 1000
            return prediction, float(prob), latency

        except Exception as exc:
            raise RuntimeError(f"Prediction failed for {self.model_name}: {exc}") from exc
