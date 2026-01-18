"""Unified model wrapper for all 7 models in V1.4 stress test suite."""
import time
import torch
import torch.nn as nn
import joblib
import numpy as np
from pathlib import Path
from typing import Tuple, Any
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.torch_models import PayloadCNN, URLCNN, TimeSeriesLSTM, MetaClassifier


class ModelWrapper:
    """Unified interface for all 7 models."""
    
    PYTORCH_MODELS = ['payload', 'url', 'timeseries', 'meta']
    SKLEARN_MODELS = ['fraud', 'host', 'network']
    
    def __init__(self, model_name: str, models_dir: Path = None):
        if model_name not in self.PYTORCH_MODELS + self.SKLEARN_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model_name = model_name
        self.models_dir = models_dir or Path(__file__).parent.parent.parent.parent / 'models'
        self.model = None
        self.scaler = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load(self) -> 'ModelWrapper':
        """Load model from disk."""
        if self.model_name in self.PYTORCH_MODELS:
            self._load_pytorch()
        else:
            self._load_sklearn()
        return self
    
    def _load_pytorch(self):
        """Load PyTorch model."""
        if self.model_name == 'payload':
            model_path = self.models_dir / 'payload_cnn.pt'
        elif self.model_name == 'url':
            model_path = self.models_dir / 'url_cnn.pt'
        elif self.model_name == 'timeseries':
            model_path = self.models_dir / 'timeseries_lstm.pt'
        elif self.model_name == 'meta':
            model_path = self.models_dir / 'meta_classifier.pt'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load TorchScript model directly
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
    
    def _load_sklearn(self):
        """Load sklearn model."""
        if self.model_name == 'fraud':
            model_path = self.models_dir / 'fraud_detection_model.pkl'
            scaler_path = self.models_dir / 'fraud_scaler.pkl'
        elif self.model_name == 'host':
            model_path = self.models_dir / 'host_behavior_model.pkl'
            scaler_path = self.models_dir / 'host_behavior_scaler.pkl'
        elif self.model_name == 'network':
            model_path = self.models_dir / 'network_intrusion_model.pkl'
            scaler_path = self.models_dir / 'network_scaler.pkl'
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        # Fix sklearn parallelism warning by using single-threaded backend
        from joblib import parallel_backend
        with parallel_backend('loky', n_jobs=1):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
    
    def preprocess(self, input_data: Any) -> Any:
        """Convert scenario input to model-ready format."""
        if self.model_name == 'payload':
            # String → char indices, pad to 500
            if isinstance(input_data, str):
                indices = [ord(c) % 256 for c in input_data[:500]]
                indices += [0] * (500 - len(indices))
                return torch.tensor([indices], dtype=torch.long, device=self.device)
        
        elif self.model_name == 'url':
            # String → char indices, pad to 200
            if isinstance(input_data, str):
                indices = [ord(c) % 128 for c in input_data[:200]]
                indices += [0] * (200 - len(indices))
                return torch.tensor([indices], dtype=torch.long, device=self.device)
        
        elif self.model_name == 'timeseries':
            # Ensure shape [1, 60, 8]
            if isinstance(input_data, np.ndarray):
                if input_data.shape == (60, 8):
                    input_data = input_data[np.newaxis, :]
                return torch.tensor(input_data, dtype=torch.float32, device=self.device)
        
        elif self.model_name == 'meta':
            # Ensure shape [1, 5]
            if isinstance(input_data, (list, np.ndarray)):
                input_data = np.array(input_data).reshape(1, -1)
                return torch.tensor(input_data, dtype=torch.float32, device=self.device)
        
        elif self.model_name in self.SKLEARN_MODELS:
            # Apply scaler, ensure 2D
            if isinstance(input_data, (list, np.ndarray)):
                input_data = np.array(input_data).reshape(1, -1)
                return self.scaler.transform(input_data)
        
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
            
            if isinstance(self.model, nn.Module):
                with torch.no_grad():
                    logits = self.model(processed)
                    prob = torch.sigmoid(logits).item()
            else:  # sklearn
                prob = self.model.predict_proba(processed)[0, 1]
            
            latency = (time.perf_counter() - start) * 1000
            prediction = 1 if prob > 0.5 else 0
            
            return (prediction, float(prob), latency)
        
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            raise RuntimeError(f"Prediction failed for {self.model_name}: {e}") from e

