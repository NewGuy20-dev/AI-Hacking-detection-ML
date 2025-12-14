"""Router to direct inputs to appropriate specialized model (4 models)."""
import joblib
import pickle
import numpy as np
import math
from pathlib import Path
from urllib.parse import urlparse


class ModelRouter:
    """Routes inputs to Network, Fraud, URL, or Payload model based on input type."""
    
    def __init__(self, models_dir: str = None):
        base = Path('/workspaces/AI-Hacking-detection-ML')
        self.models_dir = Path(models_dir) if models_dir else base / 'models'
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all specialized models."""
        model_files = {
            'network': 'network_intrusion_model.pkl',
            'fraud': 'fraud_detection_model.pkl',
            'url': 'url_analysis_model.pkl'
        }
        
        for name, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                self.models[name] = joblib.load(path)
                print(f"Loaded {name} model")
        
        # Load payload classifier
        payload_path = self.models_dir / 'payload_classifier.pkl'
        if payload_path.exists():
            with open(payload_path, 'rb') as f:
                self.models['payload'] = pickle.load(f)
            print("Loaded payload model")
    
    def detect_input_type(self, data) -> str:
        """Auto-detect input type."""
        if isinstance(data, str):
            # Check if URL
            if '://' in data or '.' in data and '/' in data:
                return 'url'
            return 'unknown'
        
        if isinstance(data, dict):
            keys = set(data.keys())
            if keys & {'url', 'domain'}:
                return 'url'
            if keys & {'amount', 'transaction', 'V1', 'V2'}:
                return 'fraud'
            if keys & {'src_ip', 'dst_ip', 'protocol', 'src_bytes', 'duration'}:
                return 'network'
        
        if isinstance(data, (list, np.ndarray)):
            if len(data) == 30:  # Credit card features
                return 'fraud'
            elif len(data) <= 10:
                return 'network'
        
        return 'network'  # Default
    
    def predict(self, data, model_type: str = None) -> dict:
        """Route to appropriate model and predict."""
        if model_type is None:
            model_type = self.detect_input_type(data)
        
        if model_type not in self.models:
            return {'error': f'Model {model_type} not loaded'}
        
        # Handle payload model separately
        if model_type == 'payload':
            m = self.models['payload']
            X = m['vectorizer'].transform([str(data)])
            proba = m['classifier'].predict_proba(X)[0]
            prob = proba[1] if len(proba) > 1 else proba[0]
            return {
                'model_type': 'payload',
                'prediction': 1 if prob > 0.5 else 0,
                'probability': float(prob),
                'is_threat': prob > 0.5,
                'confidence': float(prob) if prob > 0.5 else float(1 - prob)
            }
        
        model_data = self.models[model_type]
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Prepare features based on model type
        if model_type == 'url':
            features = self._extract_url_features(data)
        elif model_type == 'fraud':
            features = np.array(data).reshape(1, -1) if isinstance(data, list) else data
        else:  # network
            features = np.array(data).reshape(1, -1) if isinstance(data, list) else data
        
        # Scale and predict
        features_scaled = scaler.transform(features.reshape(1, -1) if features.ndim == 1 else features)
        pred = model.predict(features_scaled)
        prob = model.predict_proba(features_scaled)
        
        return {
            'model_type': model_type,
            'prediction': int(pred[0]),
            'probability': float(prob[0].max()),
            'is_threat': bool(pred[0] == 1),
            'confidence': float(prob[0][pred[0]])
        }
    
    def _extract_url_features(self, url: str) -> np.ndarray:
        """Extract features from URL."""
        try:
            parsed = urlparse(url if '://' in str(url) else f'http://{url}')
            domain = parsed.netloc or str(url).split('/')[0]
        except:
            domain = str(url)
        
        url_str = str(url)
        entropy = 0
        if url_str:
            prob = [url_str.count(c)/len(url_str) for c in set(url_str)]
            entropy = -sum(p * math.log2(p) for p in prob if p > 0)
        
        return np.array([
            len(url_str), len(domain), url_str.count('/'), url_str.count('.'),
            url_str.count('-'), sum(c.isdigit() for c in url_str) / max(len(url_str), 1),
            entropy, 1 if any(url_str.endswith(t) for t in ['.xyz','.tk','.ml','.ga','.top']) else 0
        ], dtype='float32')
    
    def predict_batch(self, data_list: list, model_type: str) -> list:
        """Batch prediction for same model type."""
        return [self.predict(d, model_type) for d in data_list]
    
    def get_metrics(self) -> dict:
        """Get training metrics for all models."""
        metrics_path = self.models_dir / 'training_metrics.pkl'
        if metrics_path.exists():
            return joblib.load(metrics_path)
        return {}


if __name__ == '__main__':
    router = ModelRouter()
    
    print("\n=== Testing Router ===")
    
    # Test URL
    url_result = router.predict('http://malicious-site.xyz/login.php', 'url')
    print(f"\nURL Test: {url_result}")
    
    # Test Network (9 features)
    network_data = [100, 500, 200, 50, 30, 0.1, 0.8, 100, 50]
    network_result = router.predict(network_data, 'network')
    print(f"Network Test: {network_result}")
    
    # Test Fraud (30 features - V1-V28 + Time + Amount)
    fraud_data = [0] * 28 + [100, 150.0]  # Simplified
    fraud_result = router.predict(fraud_data, 'fraud')
    print(f"Fraud Test: {fraud_result}")
    
    print(f"\nTraining Metrics: {router.get_metrics()}")
