"""Phase C1: sklearn model recovery/retrain script."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb


def check_model_valid(path):
    """Check if model file is valid (not LFS pointer)."""
    try:
        with open(path, 'rb') as f:
            header = f.read(20)
            if b'git-lfs' in header or b'version https' in header:
                return False
        return True
    except:
        return False


def generate_network_data(n=10000):
    """Generate synthetic network intrusion data."""
    np.random.seed(42)
    
    # Normal traffic
    normal = np.random.randn(n // 2, 41) * 0.5
    normal[:, 0] = np.random.uniform(0, 1000, n // 2)  # duration
    normal[:, 4] = np.random.uniform(0, 5000, n // 2)  # src_bytes
    normal[:, 5] = np.random.uniform(0, 5000, n // 2)  # dst_bytes
    
    # Attack traffic
    attack = np.random.randn(n // 2, 41) * 1.5
    attack[:, 0] = np.random.uniform(0, 100, n // 2)   # shorter duration
    attack[:, 4] = np.random.uniform(5000, 50000, n // 2)  # more src_bytes
    attack[:, 22] = np.random.uniform(50, 100, n // 2)  # high count
    
    X = np.vstack([normal, attack])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    
    return X, y


def generate_fraud_data(n=10000):
    """Generate synthetic fraud detection data."""
    np.random.seed(42)
    
    # Normal transactions
    normal = np.random.randn(n // 2, 30)
    normal[:, 0] = np.random.uniform(-2, 2, n // 2)  # V1
    normal[:, 28] = np.random.uniform(10, 200, n // 2)  # Amount
    
    # Fraudulent transactions
    fraud = np.random.randn(n // 2, 30) * 2
    fraud[:, 0] = np.random.uniform(-5, -2, n // 2)  # V1 anomaly
    fraud[:, 28] = np.random.uniform(500, 5000, n // 2)  # High amount
    
    X = np.vstack([normal, fraud])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    
    return X, y


def generate_url_features(n=10000):
    """Generate synthetic URL feature data."""
    np.random.seed(42)
    
    # Benign URLs
    benign = np.zeros((n // 2, 15))
    benign[:, 0] = np.random.uniform(20, 80, n // 2)   # length
    benign[:, 1] = np.random.uniform(1, 3, n // 2)     # num_dots
    benign[:, 2] = np.random.uniform(0, 1, n // 2)     # num_hyphens
    benign[:, 7] = 1  # has_https
    
    # Malicious URLs
    malicious = np.zeros((n // 2, 15))
    malicious[:, 0] = np.random.uniform(50, 200, n // 2)  # longer
    malicious[:, 1] = np.random.uniform(3, 8, n // 2)     # more dots
    malicious[:, 2] = np.random.uniform(2, 5, n // 2)     # more hyphens
    malicious[:, 6] = np.random.choice([0, 1], n // 2, p=[0.3, 0.7])  # has_ip
    malicious[:, 12] = 1  # suspicious_tld
    
    X = np.vstack([benign, malicious])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    
    return X, y


def train_network_model(models_dir):
    """Train network intrusion detection model."""
    print("\n--- Training Network Intrusion Model ---")
    X, y = generate_network_data(20000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f"  Accuracy: {acc:.2%}")
    
    joblib.dump(model, models_dir / 'network_intrusion_model.pkl')
    joblib.dump(scaler, models_dir / 'network_scaler.pkl')
    print(f"  ✓ Saved network_intrusion_model.pkl")
    return model


def train_fraud_model(models_dir):
    """Train fraud detection model."""
    print("\n--- Training Fraud Detection Model ---")
    X, y = generate_fraud_data(20000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f"  Accuracy: {acc:.2%}")
    
    joblib.dump(model, models_dir / 'fraud_detection_model.pkl')
    print(f"  ✓ Saved fraud_detection_model.pkl")
    return model


def train_url_model(models_dir):
    """Train URL analysis model."""
    print("\n--- Training URL Analysis Model ---")
    X, y = generate_url_features(20000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
    model.fit(X_train, y_train)
    
    acc = model.score(X_test, y_test)
    print(f"  Accuracy: {acc:.2%}")
    
    joblib.dump(model, models_dir / 'url_analysis_model.pkl')
    print(f"  ✓ Saved url_analysis_model.pkl")
    return model


def train_anomaly_model(models_dir):
    """Train anomaly detection model."""
    print("\n--- Training Anomaly Detector ---")
    X, _ = generate_network_data(10000)
    X_normal = X[:5000]  # Only normal data
    
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    model.fit(X_normal)
    
    joblib.dump(model, models_dir / 'anomaly_detector.pkl')
    print(f"  ✓ Saved anomaly_detector.pkl")
    return model


def main():
    base_path = Path(__file__).parent.parent
    models_dir = base_path / 'models'
    models_dir.mkdir(exist_ok=True)
    
    print("=== sklearn Model Recovery/Retrain ===")
    
    # Check existing models
    sklearn_models = [
        'network_intrusion_model.pkl',
        'fraud_detection_model.pkl',
        'url_analysis_model.pkl',
        'anomaly_detector.pkl'
    ]
    
    needs_retrain = []
    for model_name in sklearn_models:
        path = models_dir / model_name
        if not path.exists():
            print(f"  ✗ {model_name}: Not found")
            needs_retrain.append(model_name)
        elif not check_model_valid(path):
            print(f"  ✗ {model_name}: LFS pointer (invalid)")
            needs_retrain.append(model_name)
        else:
            try:
                joblib.load(path)
                print(f"  ✓ {model_name}: Valid")
            except Exception as e:
                print(f"  ✗ {model_name}: Load error - {e}")
                needs_retrain.append(model_name)
    
    if not needs_retrain:
        print("\nAll sklearn models are valid!")
        return
    
    print(f"\nRetraining {len(needs_retrain)} models...")
    
    # Retrain needed models
    if 'network_intrusion_model.pkl' in needs_retrain:
        train_network_model(models_dir)
    
    if 'fraud_detection_model.pkl' in needs_retrain:
        train_fraud_model(models_dir)
    
    if 'url_analysis_model.pkl' in needs_retrain:
        train_url_model(models_dir)
    
    if 'anomaly_detector.pkl' in needs_retrain:
        train_anomaly_model(models_dir)
    
    print("\n=== Recovery Complete ===")
    print("All sklearn models are now ready for use.")


if __name__ == "__main__":
    main()
