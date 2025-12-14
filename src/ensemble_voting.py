"""Ensemble voting classifier combining multiple detection models."""
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb


class EnsembleVoting:
    """Soft voting ensemble of RF, XGBoost, LightGBM."""
    
    def __init__(self):
        self.model = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=4, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=6, n_jobs=4, random_state=42, eval_metric='logloss')),
                ('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=15, n_jobs=4, random_state=42, verbose=-1))
            ],
            voting='soft',
            n_jobs=4
        )
        self.scaler = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    @classmethod
    def load(cls, path: str) -> 'EnsembleVoting':
        data = joblib.load(path)
        ens = cls()
        ens.model = data['model']
        ens.scaler = data['scaler']
        return ens


def train_ensemble():
    from train_unified import load_nsl_unified, load_cicids_unified, UNIFIED_FEATURES
    import pandas as pd
    import gc
    
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    print("Loading data for ensemble training...")
    df_nsl_train = load_nsl_unified(str(base / 'data/nsl_kdd_train.csv'))
    df_nsl_test = load_nsl_unified(str(base / 'data/nsl_kdd_test.csv'))
    df_nsl = pd.concat([df_nsl_train, df_nsl_test], ignore_index=True)
    df_cicids = load_cicids_unified(str(base / 'cicids2017/MachineLearningCVE'), sample_per_file=15000)
    
    df_all = pd.concat([df_nsl, df_cicids], ignore_index=True)
    del df_nsl, df_cicids
    gc.collect()
    
    X = df_all[UNIFIED_FEATURES].astype('float32').values
    y = df_all['is_attack'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training ensemble on {len(X_train)} samples...")
    ensemble = EnsembleVoting()
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    print(f"\nEnsemble Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    ensemble.save(str(base / 'models/ensemble_voting.pkl'))
    # Also save scaler separately for validation
    joblib.dump(ensemble.scaler, base / 'models/ensemble_scaler.pkl')
    print("Ensemble saved!")
    return ensemble


if __name__ == '__main__':
    train_ensemble()
