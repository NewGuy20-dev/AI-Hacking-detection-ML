"""Router to direct inputs to appropriate specialized model (4 models)."""
import joblib
import pickle
import numpy as np
import math
import json
from pathlib import Path
from urllib.parse import urlparse

try:
    from confidence import ConfidenceCalibrator
except Exception:
    ConfidenceCalibrator = None

try:
    from benign_filter import get_filter
except Exception:
    get_filter = None


class ModelRouter:
    """Routes inputs to Network, Fraud, URL, or Payload model based on input type."""
    
    def __init__(self, models_dir: str = None):
        default_root = Path(__file__).resolve().parents[1]
        self.models_dir = Path(models_dir) if models_dir else default_root / 'models'
        self.models = {}
        self.thresholds = self._load_thresholds()
        self.calibrators = {}
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
                self._load_calibrator(name)
        
        # Load payload classifier
        payload_path = self.models_dir / 'payload_classifier.pkl'
        if payload_path.exists():
            with open(payload_path, 'rb') as f:
                self.models['payload'] = pickle.load(f)
            print("Loaded payload model")
            self._load_calibrator('payload')

    def _load_calibrator(self, model_name: str):
        if ConfidenceCalibrator is None:
            return
        cal_path = self.models_dir / 'calibration' / f"{model_name}_calibration.json"
        if cal_path.exists():
            cal = ConfidenceCalibrator()
            cal.load(cal_path)
            self.calibrators[model_name] = cal

    def _load_thresholds(self) -> dict:
        config_path = Path(__file__).resolve().parents[1] / 'configs' / 'inference' / 'optimal_thresholds.json'
        thresholds = {'default': 0.5}
        if config_path.exists():
            try:
                data = json.loads(config_path.read_text(encoding='utf-8'))
                if isinstance(data, dict) and isinstance(data.get('thresholds'), dict):
                    thresholds.update(data['thresholds'])
            except Exception:
                pass
        return thresholds

    def _get_threshold(self, model_type: str) -> float:
        mapping = {
            'payload': 'payload_cnn',
            'url': 'url_cnn',
            'network': 'network',
            'fraud': 'fraud',
            'host': 'host',
        }
        key = mapping.get(model_type, model_type)
        return float(self.thresholds.get(key, self.thresholds.get('default', 0.5)))
    
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
            if get_filter is not None and isinstance(data, str):
                prefilter = get_filter()
                is_benign, benign_conf, reason = prefilter.is_benign(data)
                if is_benign:
                    attack_prob = max(0.0, 1.0 - float(benign_conf))
                    return {
                        'model_type': 'payload',
                        'prediction': 0,
                        'probability': float(attack_prob),
                        'is_threat': False,
                        'confidence': float(1.0 - attack_prob),
                        'prefilter_reason': reason,
                        'prefiltered': True,
                    }
            m = self.models['payload']
            X = m['vectorizer'].transform([str(data)])
            proba = m['classifier'].predict_proba(X)[0]
            prob = proba[1] if len(proba) > 1 else proba[0]
            if model_type in self.calibrators:
                prob = float(self.calibrators[model_type].calibrate(np.array([prob]))[0])
            threshold = self._get_threshold('payload')
            return {
                'model_type': 'payload',
                'prediction': 1 if prob > threshold else 0,
                'probability': float(prob),
                'is_threat': prob > threshold,
                'confidence': float(prob) if prob > threshold else float(1 - prob),
                'threshold_used': threshold,
                'prefiltered': False,
            }
        
        model_data = self.models[model_type]
        model = model_data['model'] if isinstance(model_data, dict) and 'model' in model_data else model_data
        scaler = model_data.get('scaler') if isinstance(model_data, dict) else None
        
        # Prepare features based on model type
        if model_type == 'url':
            features = self._extract_url_features(data)
        elif model_type == 'fraud':
            features = np.array(data).reshape(1, -1) if isinstance(data, list) else data
        else:  # network
            features = np.array(data).reshape(1, -1) if isinstance(data, list) else data
        
        # Scale and predict
        if scaler is not None:
            features_scaled = scaler.transform(features.reshape(1, -1) if features.ndim == 1 else features)
        else:
            features_scaled = features.reshape(1, -1) if features.ndim == 1 else features
        prob_vec = model.predict_proba(features_scaled)
        prob = float(prob_vec[0, 1]) if prob_vec.shape[1] > 1 else float(prob_vec[0, 0])
        if model_type in self.calibrators:
            prob = float(self.calibrators[model_type].calibrate(np.array([prob]))[0])
        threshold = self._get_threshold(model_type)
        pred = int(prob > threshold)
        
        return {
            'model_type': model_type,
            'prediction': int(pred),
            'probability': float(prob),
            'is_threat': bool(pred == 1),
            'confidence': float(prob) if pred == 1 else float(1 - prob),
            'threshold_used': threshold,
        }
    
    def _extract_url_features(self, url: str) -> np.ndarray:
        """Extract features from URL."""
        try:
            parsed = urlparse(url if '://' in str(url) else f'http://{url}')
            domain = parsed.netloc or str(url).split('/')[0]
        except:
            domain = str(url)
        
        url_str = str(url)
        
        # Must match training features (10 features)
        return np.array([
            len(url_str), url_str.count('/'), url_str.count('.'), url_str.count('-'),
            url_str.count('?'), url_str.count('='), url_str.count('&'),
            sum(c.isdigit() for c in url_str) / max(len(url_str), 1),
            sum(c.isupper() for c in url_str) / max(len(url_str), 1),
            1 if any(url_str.endswith(t) for t in ['.xyz','.tk','.ml','.ga','.top','.pw']) else 0
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
