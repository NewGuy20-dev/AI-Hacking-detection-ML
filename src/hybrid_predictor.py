"""HybridPredictor: Combines sklearn (CPU) and PyTorch (GPU) models for inference."""
import torch
import joblib
import numpy as np
from pathlib import Path


class HybridPredictor:
    """Unified predictor combining sklearn and PyTorch models."""
    
    def __init__(self, models_dir='models', device=None):
        self.models_dir = Path(models_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model containers
        self.sklearn_models = {}
        self.pytorch_models = {}
        self.loaded = False
    
    def load_models(self):
        """Load all available models."""
        print(f"Loading models from {self.models_dir}...")
        print(f"Device: {self.device}")
        
        # sklearn models (CPU)
        sklearn_files = {
            'network': 'network_intrusion_model.pkl',
            'fraud': 'fraud_detection_model.pkl',
            'url_lgbm': 'url_analysis_model.pkl',
            'anomaly': 'anomaly_detector.pkl',
        }
        
        for name, filename in sklearn_files.items():
            path = self.models_dir / filename
            if path.exists():
                try:
                    self.sklearn_models[name] = joblib.load(path)
                    print(f"  ✓ {name} (sklearn)")
                except Exception as e:
                    print(f"  ✗ {name}: {e}")
        
        # PyTorch models (GPU)
        pytorch_files = {
            'payload_cnn': 'payload_cnn.pt',
            'url_cnn': 'url_cnn.pt',
            'timeseries_lstm': 'timeseries_lstm.pt',
            'meta_classifier': 'meta_classifier.pt',
        }
        
        for name, filename in pytorch_files.items():
            path = self.models_dir / filename
            if path.exists():
                try:
                    model = torch.jit.load(str(path), map_location=self.device)
                    model.eval()
                    self.pytorch_models[name] = model
                    print(f"  ✓ {name} (PyTorch)")
                except Exception as e:
                    print(f"  ✗ {name}: {e}")
        
        self.loaded = True
        print(f"Loaded: {len(self.sklearn_models)} sklearn, {len(self.pytorch_models)} PyTorch")
        return self
    
    def _tokenize_payload(self, text, max_len=500):
        """Convert text to character indices."""
        chars = [ord(c) % 256 for c in str(text)[:max_len]]
        chars += [0] * (max_len - len(chars))
        return chars
    
    def _tokenize_url(self, url, max_len=200):
        """Convert URL to character indices."""
        chars = [ord(c) % 128 for c in str(url)[:max_len]]
        chars += [0] * (max_len - len(chars))
        return chars
    
    @torch.no_grad()
    def predict_payload(self, payloads):
        """Predict malicious probability for payloads."""
        if 'payload_cnn' not in self.pytorch_models:
            return np.full(len(payloads), 0.5)
        
        tokens = [self._tokenize_payload(p) for p in payloads]
        x = torch.tensor(tokens, dtype=torch.long).to(self.device)
        
        with torch.amp.autocast('cuda'):
            logits = self.pytorch_models['payload_cnn'](x)
        
        return torch.sigmoid(logits).cpu().numpy()
    
    @torch.no_grad()
    def predict_url(self, urls):
        """Predict malicious probability for URLs."""
        if 'url_cnn' not in self.pytorch_models:
            return np.full(len(urls), 0.5)
        
        tokens = [self._tokenize_url(u) for u in urls]
        x = torch.tensor(tokens, dtype=torch.long).to(self.device)
        
        with torch.amp.autocast('cuda'):
            logits = self.pytorch_models['url_cnn'](x)
        
        return torch.sigmoid(logits).cpu().numpy()
    
    @torch.no_grad()
    def predict_timeseries(self, sequences):
        """Predict anomaly probability for time-series data."""
        if 'timeseries_lstm' not in self.pytorch_models:
            return np.full(len(sequences), 0.5)
        
        x = torch.tensor(sequences, dtype=torch.float32).to(self.device)
        
        with torch.amp.autocast('cuda'):
            logits = self.pytorch_models['timeseries_lstm'](x)
        
        return torch.sigmoid(logits).cpu().numpy()
    
    @torch.no_grad()
    def predict_ensemble(self, model_scores):
        """Combine model scores using meta-classifier."""
        if 'meta_classifier' not in self.pytorch_models:
            # Fallback to weighted average
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
            return np.average(model_scores, axis=1, weights=weights)
        
        x = torch.tensor(model_scores, dtype=torch.float32).to(self.device)
        
        with torch.amp.autocast('cuda'):
            logits = self.pytorch_models['meta_classifier'](x)
        
        return torch.sigmoid(logits).cpu().numpy()
    
    def predict(self, data):
        """
        Full prediction pipeline.
        
        Args:
            data: dict with optional keys:
                - 'payloads': list of payload strings
                - 'urls': list of URL strings
                - 'timeseries': array of shape (n, seq_len, features)
                - 'network_features': array for sklearn network model
                - 'fraud_features': array for sklearn fraud model
        
        Returns:
            dict with predictions and component scores
        """
        if not self.loaded:
            self.load_models()
        
        n_samples = 1
        scores = {}
        
        # PyTorch predictions
        if 'payloads' in data:
            payloads = data['payloads'] if isinstance(data['payloads'], list) else [data['payloads']]
            n_samples = len(payloads)
            scores['payload'] = self.predict_payload(payloads)
        
        if 'urls' in data:
            urls = data['urls'] if isinstance(data['urls'], list) else [data['urls']]
            n_samples = max(n_samples, len(urls))
            scores['url'] = self.predict_url(urls)
        
        if 'timeseries' in data:
            ts = np.array(data['timeseries'])
            if ts.ndim == 2:
                ts = ts[np.newaxis, ...]
            n_samples = max(n_samples, len(ts))
            scores['timeseries'] = self.predict_timeseries(ts)
        
        # Fill missing scores with 0.5 (neutral)
        for key in ['payload', 'url', 'timeseries', 'network', 'fraud']:
            if key not in scores:
                scores[key] = np.full(n_samples, 0.5)
        
        # Combine using meta-classifier
        model_scores = np.column_stack([
            scores['network'],
            scores['fraud'],
            scores['url'],
            scores['payload'],
            scores['timeseries']
        ])
        
        final_prob = self.predict_ensemble(model_scores)
        
        return {
            'is_attack': (final_prob > 0.5).astype(int),
            'confidence': final_prob,
            'scores': scores
        }


def create_predictor(models_dir='models'):
    """Factory function to create and load predictor."""
    predictor = HybridPredictor(models_dir)
    predictor.load_models()
    return predictor
