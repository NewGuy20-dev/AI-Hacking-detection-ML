"""Ensemble model combining Network, URL, and Content detectors with calibration."""
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
from typing import Dict, Optional, List

try:
    from confidence import EnsembleCalibrator, ConfidenceCalibrator
    from threshold_optimizer import load_optimal_thresholds
except ImportError:
    EnsembleCalibrator = None
    load_optimal_thresholds = None

try:
    from context_classifier import ContextAwareClassifier, quick_adjust
    HAS_CONTEXT = True
except ImportError:
    HAS_CONTEXT = False
    quick_adjust = None


class EnsembleDetector:
    """Ensemble detector combining multiple specialized models with calibration."""
    
    DEFAULT_WEIGHTS = {'network': 0.5, 'url': 0.3, 'content': 0.2}
    
    def __init__(self, models_dir: str = None, use_meta_classifier: bool = False,
                 use_context_aware: bool = True):
        self.models_dir = Path(models_dir) if models_dir else None
        self.network_model = None
        self.url_model = None
        self.content_model = None
        self.meta_classifier = None
        self.use_meta_classifier = use_meta_classifier
        self.weights = self.DEFAULT_WEIGHTS.copy()
        self.calibrator = EnsembleCalibrator() if EnsembleCalibrator else None
        self.thresholds = {"default": 0.5}
        self.use_calibration = False
        self.use_context_aware = use_context_aware and HAS_CONTEXT
        self.context_classifier = ContextAwareClassifier() if self.use_context_aware else None
    
    def load_models(self):
        """Load all detector models."""
        if not self.models_dir:
            raise ValueError("models_dir not set")
        
        network_path = self.models_dir / 'network_detector.pkl'
        url_path = self.models_dir / 'url_detector.pkl'
        content_path = self.models_dir / 'content_detector.pkl'
        
        if network_path.exists():
            self.network_model = joblib.load(network_path)
        if url_path.exists():
            self.url_model = joblib.load(url_path)
        if content_path.exists():
            self.content_model = joblib.load(content_path)
        
        # Load calibration if available
        cal_dir = self.models_dir / 'calibration'
        if cal_dir.exists() and self.calibrator:
            self.calibrator.load(cal_dir)
            self.use_calibration = True
        
        # Load optimal thresholds if available
        thresh_path = self.models_dir.parent / 'configs' / 'optimal_thresholds.json'
        if thresh_path.exists() and load_optimal_thresholds:
            self.thresholds = load_optimal_thresholds(thresh_path)
        
        return self
    
    def predict_proba_network(self, X: np.ndarray) -> np.ndarray:
        """Get attack probability from network detector."""
        if self.network_model is None:
            return np.full(len(X), 0.5)
        probs = self.network_model.predict_proba(X)[:, 1]
        if self.use_calibration and self.calibrator:
            probs = self.calibrator.calibrate_model('network', probs)
        return probs
    
    def predict_proba_url(self, X: np.ndarray) -> np.ndarray:
        """Get malicious probability from URL detector."""
        if self.url_model is None:
            return np.full(len(X), 0.5)
        probs = self.url_model.predict_proba(X)[:, 1]
        if self.use_calibration and self.calibrator:
            probs = self.calibrator.calibrate_model('url', probs)
        return probs
    
    def predict_proba_content(self, X: np.ndarray) -> np.ndarray:
        """Get phishing/spam probability from content detector."""
        if self.content_model is None:
            return np.full(len(X), 0.5)
        probs = self.content_model.predict_proba(X)[:, 1]
        if self.use_calibration and self.calibrator:
            probs = self.calibrator.calibrate_model('content', probs)
        return probs
    
    def weighted_average(self, probs: dict) -> np.ndarray:
        """Combine probabilities using weighted average."""
        total_weight = sum(self.weights[k] for k in probs.keys())
        combined = sum(probs[k] * self.weights[k] for k in probs.keys())
        return combined / total_weight
    
    def stacking_predict(self, probs: dict) -> np.ndarray:
        """Use stacking meta-classifier for final prediction."""
        if not self.meta_classifier:
            return self.weighted_average(probs)
        
        X = np.column_stack([probs.get('network', np.zeros(1)), 
                            probs.get('url', np.zeros(1)), 
                            probs.get('content', np.zeros(1))])
        return self.meta_classifier.predict_proba(X)[:, 1]
    
    def fit_meta_classifier(self, X_network: np.ndarray, X_url: np.ndarray, 
                           X_content: np.ndarray, y: np.ndarray):
        """Train meta-classifier on detector outputs (stacking)."""
        probs = np.column_stack([
            self.predict_proba_network(X_network),
            self.predict_proba_url(X_url),
            self.predict_proba_content(X_content)
        ])
        
        self.meta_classifier = LogisticRegression(random_state=42, C=1.0)
        self.meta_classifier.fit(probs, y)
        self.use_meta_classifier = True
        return self
    
    def fit_calibration(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]):
        """Fit calibration for each model."""
        if not self.calibrator:
            return
        
        for model_name, y_prob in predictions.items():
            self.calibrator.fit_model(model_name, y_true, y_prob, method="platt")
        
        self.use_calibration = True
    
    def get_threshold(self, model_type: str = "default") -> float:
        """Get optimal threshold for model type."""
        return self.thresholds.get(model_type, self.thresholds.get("default", 0.5))
    
    def apply_context_adjustment(self, texts: List[str], scores: np.ndarray) -> np.ndarray:
        """Apply context-aware adjustment to scores."""
        if not self.use_context_aware or not self.context_classifier:
            return scores
        
        adjusted = []
        for text, score in zip(texts, scores):
            result = self.context_classifier.adjust_score(text, float(score))
            adjusted.append(result.adjusted_score)
        return np.array(adjusted)
    
    def predict(self, network_probs: np.ndarray = None, url_probs: np.ndarray = None,
                content_probs: np.ndarray = None, threshold: float = None,
                return_breakdown: bool = False, texts: List[str] = None) -> dict:
        """Make final prediction combining available detector outputs."""
        probs = {}
        if network_probs is not None:
            probs['network'] = network_probs
        if url_probs is not None:
            probs['url'] = url_probs
        if content_probs is not None:
            probs['content'] = content_probs
        
        if not probs:
            raise ValueError("At least one detector output required")
        
        # Use threshold from config if not specified
        if threshold is None:
            threshold = self.get_threshold("ensemble")
        
        # Combine predictions
        if self.use_meta_classifier and self.meta_classifier and len(probs) == 3:
            confidence = self.stacking_predict(probs)
        else:
            confidence = self.weighted_average(probs)
        
        # Apply context-aware adjustment if texts provided
        if texts is not None and self.use_context_aware:
            confidence = self.apply_context_adjustment(texts, confidence)
        
        result = {
            'is_attack': (confidence >= threshold).astype(int),
            'confidence': confidence,
            'threshold_used': threshold,
            'context_aware': texts is not None and self.use_context_aware,
        }
        
        if return_breakdown:
            result['model_scores'] = probs
            result['weights'] = self.weights
        
        return result
    
    def save(self, path: str):
        """Save ensemble model."""
        joblib.dump(self, path)
        
        # Save calibration separately
        if self.calibrator and self.models_dir:
            cal_dir = self.models_dir / 'calibration'
            self.calibrator.save(cal_dir)
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleDetector':
        """Load ensemble model."""
        return joblib.load(path)


def evaluate_ensemble(ensemble: EnsembleDetector, X_network: np.ndarray, 
                     X_url: np.ndarray, X_content: np.ndarray, y: np.ndarray):
    """Evaluate ensemble performance."""
    network_probs = ensemble.predict_proba_network(X_network)
    url_probs = ensemble.predict_proba_url(X_url)
    content_probs = ensemble.predict_proba_content(X_content)
    
    result = ensemble.predict(network_probs, url_probs, content_probs)
    
    print(f"Accuracy: {accuracy_score(y, result['is_attack']):.4f}")
    print(f"F1 Score: {f1_score(y, result['is_attack']):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, result['is_attack'], target_names=['Normal', 'Attack']))


if __name__ == '__main__':
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Create and save ensemble
    print("Creating ensemble detector...")
    ensemble = EnsembleDetector(models_dir=str(base / 'models'))
    ensemble.load_models()
    
    print(f"Network model loaded: {ensemble.network_model is not None}")
    print(f"URL model loaded: {ensemble.url_model is not None}")
    print(f"Content model loaded: {ensemble.content_model is not None}")
    print(f"Weights: {ensemble.weights}")
    
    ensemble.save(str(base / 'models/ensemble.pkl'))
    print(f"\nEnsemble saved to models/ensemble.pkl")
