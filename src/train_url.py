"""Train URL Detector using LightGBM."""
import gc
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
import lightgbm as lgb

from feature_engineering import extract_url_features_batch, get_url_feature_names


def load_url_data(base_path: Path):
    """Load and combine URL datasets."""
    urls, labels = [], []
    
    # Load phishing dataset (ARFF)
    phishing_path = base_path / 'datasets/phishing/phishing_dataset.arff'
    if phishing_path.exists():
        with open(phishing_path, 'r') as f:
            in_data = False
            for line in f:
                line = line.strip()
                if line.lower() == '@data':
                    in_data = True
                    continue
                if in_data and line and not line.startswith('%'):
                    parts = line.split(',')
                    # Use features to generate synthetic URL-like data
                    urls.append(f"http://site{len(urls)}.com/page{parts[0]}")
                    labels.append(1 if parts[-1] == '1' else 0)
    
    # Load benign domains from top-1m if available
    domains_path = base_path / 'datasets/domains/top-1m.csv'
    if domains_path.exists():
        try:
            df = pd.read_csv(domains_path, header=None, nrows=5000)
            for _, row in df.iterrows():
                domain = str(row.iloc[-1]) if len(row) > 1 else str(row.iloc[0])
                urls.append(f"https://{domain}")
                labels.append(0)
        except:
            pass
    
    # Generate synthetic malicious URLs for training
    malicious_patterns = [
        'http://192.168.1.{}/admin/login.php',
        'http://free-money-{}.xyz/claim.html',
        'http://verify-account-{}.tk/secure',
        'http://update-{}.ml/download.exe',
        'http://login-{}.ga/bank/verify',
    ]
    for i in range(2000):
        pattern = malicious_patterns[i % len(malicious_patterns)]
        urls.append(pattern.format(i))
        labels.append(1)
    
    # Add benign URLs
    benign_patterns = [
        'https://www.google.com/search?q={}',
        'https://github.com/user/repo{}',
        'https://stackoverflow.com/questions/{}',
        'https://www.amazon.com/product/{}',
        'https://en.wikipedia.org/wiki/Article{}',
    ]
    for i in range(2000):
        pattern = benign_patterns[i % len(benign_patterns)]
        urls.append(pattern.format(i))
        labels.append(0)
    
    return urls, np.array(labels)


def train_url_detector(X: np.ndarray, y: np.ndarray, save_path: str = None):
    """Train LightGBM classifier for URL detection."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = lgb.LGBMClassifier(
        n_estimators=100,
        num_leaves=31,
        max_depth=15,
        learning_rate=0.1,
        n_jobs=4,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    
    print("Training LightGBM...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
    
    # Feature importance
    print("\nTop 5 Important Features:")
    feature_names = get_url_feature_names()
    importance = sorted(zip(feature_names, model.feature_importances_), 
                       key=lambda x: x[1], reverse=True)
    for name, imp in importance[:5]:
        print(f"  {name}: {imp:.4f}")
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"\nModel saved to {save_path}")
    
    return model


if __name__ == '__main__':
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Load data
    print("Loading URL data...")
    urls, y = load_url_data(base)
    print(f"Total URLs: {len(urls)}")
    
    # Extract features
    print("Extracting URL features...")
    X = extract_url_features_batch(urls)
    del urls
    gc.collect()
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Malicious ratio: {y.mean():.2%}")
    
    # Train
    model = train_url_detector(
        X, y,
        save_path=str(base / 'models/url_detector.pkl')
    )
