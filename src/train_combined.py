"""Train Network Detector on combined NSL-KDD + CICIDS2017 datasets."""
import gc
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

from data_loader import (load_nsl_kdd, preprocess_network_data, 
                         load_cicids2017, preprocess_cicids_data)


def train_combined():
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Load NSL-KDD
    print("Loading NSL-KDD...")
    df_nsl = load_nsl_kdd(
        str(base / 'data/nsl_kdd_train.csv'),
        str(base / 'data/nsl_kdd_test.csv')
    )
    X_nsl, y_nsl, _, _ = preprocess_network_data(df_nsl)
    print(f"  NSL-KDD: {X_nsl.shape[0]} samples, {X_nsl.shape[1]} features")
    del df_nsl
    gc.collect()
    
    # Load CICIDS2017
    print("\nLoading CICIDS2017...")
    df_cicids = load_cicids2017(str(base / 'cicids2017/MachineLearningCVE'), sample_per_file=15000)
    X_cicids, y_cicids, _, _ = preprocess_cicids_data(df_cicids)
    print(f"  CICIDS2017: {X_cicids.shape[0]} samples, {X_cicids.shape[1]} features")
    del df_cicids
    gc.collect()
    
    # Combine (use common feature count - pad/truncate)
    min_features = min(X_nsl.shape[1], X_cicids.shape[1])
    X_nsl = X_nsl[:, :min_features]
    X_cicids = X_cicids[:, :min_features]
    
    X = np.vstack([X_nsl, X_cicids])
    y = np.concatenate([y_nsl, y_cicids])
    del X_nsl, X_cicids, y_nsl, y_cicids
    gc.collect()
    
    print(f"\nCombined: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Attack ratio: {y.mean():.2%}")
    
    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        n_jobs=4,
        random_state=42,
        class_weight='balanced'
    )
    
    print("\nTraining RandomForest on combined data...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Save
    joblib.dump(model, base / 'models/network_detector.pkl')
    print("\nModel saved to models/network_detector.pkl")


if __name__ == '__main__':
    train_combined()
