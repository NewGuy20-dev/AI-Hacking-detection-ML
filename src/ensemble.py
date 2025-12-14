"""Ensemble model combining Network, URL, and Content detectors."""
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score


class EnsembleDetector:
    """Ensemble detector combining multiple specialized models."""
    
    # Default weights per DEVELOPMENT_PLAN.md
    DEFAULT_WEIGHTS = {'network': 0.5, 'url': 0.3, 'content': 0.2}
    
    def __init__(self, models_dir: str = None, use_meta_classifier: bool = False):
        self.models_dir = Path(models_dir) if models_dir else None
        self.network_model = None
        self.url_model = None
        self.content_model = None
        self.meta_classifier = None
        self.use_meta_classifier = use_meta_classifier
        self.weights = self.DEFAULT_WEIGHTS.copy()
    
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
        
        return self
    
    def predict_proba_network(self, X: np.ndarray) -> np.ndarray:
        """Get attack probability from network detector."""
        if self.network_model is None:
            return np.full(len(X), 0.5)
        return self.network_model.predict_proba(X)[:, 1]
    
    def predict_proba_url(self, X: np.ndarray) -> np.ndarray:
        """Get malicious probability from URL detector."""
        if self.url_model is None:
            return np.full(len(X), 0.5)
        return self.url_model.predict_proba(X)[:, 1]
    
    def predict_proba_content(self, X: np.ndarray) -> np.ndarray:
        """Get phishing/spam probability from content detector."""
        if self.content_model is None:
            return np.full(len(X), 0.5)
        return self.content_model.predict_proba(X)[:, 1]
    
    def weighted_average(self, probs: dict) -> np.ndarray:
        """Combine probabilities using weighted average."""
        total_weight = sum(self.weights[k] for k in probs.keys())
        combined = sum(probs[k] * self.weights[k] for k in probs.keys())
        return combined / total_weight
    
    def fit_meta_classifier(self, X_network: np.ndarray, X_url: np.ndarray, 
                           X_content: np.ndarray, y: np.ndarray):
        """Train meta-classifier on detector outputs."""
        probs = np.column_stack([
            self.predict_proba_network(X_network),
            self.predict_proba_url(X_url),
            self.predict_proba_content(X_content)
        ])
        
        self.meta_classifier = LogisticRegression(random_state=42)
        self.meta_classifier.fit(probs, y)
        self.use_meta_classifier = True
        return self
    
    def predict(self, network_probs: np.ndarray = None, url_probs: np.ndarray = None,
                content_probs: np.ndarray = None, threshold: float = 0.5) -> dict:
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
        
        if self.use_meta_classifier and self.meta_classifier and len(probs) == 3:
            X = np.column_stack([probs['network'], probs['url'], probs['content']])
            confidence = self.meta_classifier.predict_proba(X)[:, 1]
        else:
            confidence = self.weighted_average(probs)
        
        return {
            'is_attack': (confidence >= threshold).astype(int),
            'confidence': confidence,
        }
    
    def save(self, path: str):
        """Save ensemble model."""
        joblib.dump(self, path)
    
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
