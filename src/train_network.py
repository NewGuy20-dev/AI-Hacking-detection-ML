"""Train Network Detector using RandomForest on NSL-KDD data."""
import gc
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score

from data_loader import load_nsl_kdd, preprocess_network_data


def train_network_detector(X: np.ndarray, y: np.ndarray, save_path: str = None):
    """Train RandomForest classifier for network intrusion detection."""
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
    
    print("Training RandomForest...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Cross-validation
    print("\n5-Fold Cross-Validation:")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=4)
    print(f"F1 Scores: {cv_scores}")
    print(f"Mean F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"\nModel saved to {save_path}")
    
    return model


if __name__ == '__main__':
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Load data
    print("Loading NSL-KDD dataset...")
    df = load_nsl_kdd(
        str(base / 'data/nsl_kdd_train.csv'),
        str(base / 'data/nsl_kdd_test.csv')
    )
    print(f"Total records: {len(df)}")
    
    # Preprocess
    print("Preprocessing...")
    X, y, scaler, feature_cols = preprocess_network_data(df)
    del df
    gc.collect()
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Attack ratio: {y.mean():.2%}")
    
    # Train
    model = train_network_detector(
        X, y, 
        save_path=str(base / 'models/network_detector.pkl')
    )
    
    # Save scaler
    joblib.dump(scaler, base / 'models/network_scaler.pkl')
    print("Scaler saved")
