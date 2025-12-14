"""Anomaly detection for zero-day attack identification."""
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler


class AnomalyDetector:
    """Isolation Forest based anomaly detector trained on benign traffic."""
    
    def __init__(self, contamination=0.01):
        self.model = IsolationForest(
            n_estimators=100, contamination=contamination,
            random_state=42, n_jobs=4
        )
        self.scaler = None
        self.fitted = False
    
    def fit(self, X_benign: np.ndarray):
        """Fit on benign traffic only."""
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X_benign)
        self.model.fit(X_scaled)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> dict:
        """Return anomaly scores and predictions."""
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)  # Higher = more anomalous
        preds = self.model.predict(X_scaled)  # -1 = anomaly, 1 = normal
        return {
            'anomaly_score': scores,
            'is_anomaly': (preds == -1).astype(int)
        }
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        data = joblib.load(path)
        detector = cls()
        detector.model = data['model']
        detector.scaler = data['scaler']
        detector.fitted = True
        return detector


def train_anomaly_detector():
    from train_unified import load_nsl_unified, load_cicids_unified, UNIFIED_FEATURES
    import pandas as pd
    
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Load benign samples only
    print("Loading benign traffic for anomaly detector...")
    df_nsl = load_nsl_unified(str(base / 'data/nsl_kdd_train.csv'))
    df_cicids = load_cicids_unified(str(base / 'cicids2017/MachineLearningCVE'), sample_per_file=10000)
    
    df_benign = pd.concat([
        df_nsl[df_nsl['is_attack'] == 0],
        df_cicids[df_cicids['is_attack'] == 0]
    ])
    
    X_benign = df_benign[UNIFIED_FEATURES].astype('float32').values
    print(f"Training on {len(X_benign)} benign samples...")
    
    detector = AnomalyDetector(contamination=0.01)
    detector.fit(X_benign)
    detector.save(str(base / 'models/anomaly_detector.pkl'))
    print("Anomaly detector saved!")
    return detector


if __name__ == '__main__':
    train_anomaly_detector()
