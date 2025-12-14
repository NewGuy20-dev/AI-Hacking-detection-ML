"""Online learning for continuous model updates without full retraining."""
import joblib
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


class OnlineLearner:
    """Incremental learning classifier using SGD."""
    
    def __init__(self):
        self.model = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=0.0001,
            random_state=42, warm_start=True
        )
        self.scaler = MinMaxScaler()
        self.fitted = False
        self.update_count = 0
        self.last_update = None
    
    def initial_fit(self, X: np.ndarray, y: np.ndarray):
        """Initial training on batch data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True
        self.update_count = 1
        self.last_update = datetime.now().isoformat()
        return self
    
    def partial_update(self, X: np.ndarray, y: np.ndarray):
        """Incremental update with new samples."""
        if not self.fitted:
            return self.initial_fit(X, y)
        
        X_scaled = self.scaler.transform(X)
        self.model.partial_fit(X_scaled, y)
        self.update_count += 1
        self.last_update = datetime.now().isoformat()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, path: str):
        joblib.dump({
            'model': self.model, 'scaler': self.scaler,
            'update_count': self.update_count, 'last_update': self.last_update
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'OnlineLearner':
        data = joblib.load(path)
        learner = cls()
        learner.model = data['model']
        learner.scaler = data['scaler']
        learner.update_count = data['update_count']
        learner.last_update = data['last_update']
        learner.fitted = True
        return learner
    
    def status(self) -> dict:
        return {
            'fitted': self.fitted,
            'update_count': self.update_count,
            'last_update': self.last_update
        }


if __name__ == '__main__':
    from sklearn.metrics import accuracy_score, f1_score
    
    # Simulate streaming data
    np.random.seed(42)
    
    # Initial batch
    X_init = np.random.randn(1000, 15)
    y_init = (X_init[:, 0] + X_init[:, 1] > 0).astype(int)
    
    learner = OnlineLearner()
    learner.initial_fit(X_init, y_init)
    print(f"Initial fit: {learner.status()}")
    
    # Simulate incremental updates
    for i in range(5):
        X_new = np.random.randn(100, 15)
        y_new = (X_new[:, 0] + X_new[:, 1] > 0).astype(int)
        learner.partial_update(X_new, y_new)
    
    print(f"After updates: {learner.status()}")
    
    # Test
    X_test = np.random.randn(200, 15)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
    y_pred = learner.predict(X_test)
    
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1: {f1_score(y_test, y_pred):.4f}")
