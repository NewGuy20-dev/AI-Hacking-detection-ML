"""Train Content Detector using XGBoost for phishing/spam detection."""
import gc
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb

from feature_engineering import ContentFeatureExtractor


def load_content_data(base_path: Path):
    """Load content/text data for training."""
    texts, labels = [], []
    
    # Phishing email patterns
    phishing_templates = [
        "URGENT: Your account has been suspended. Click here to verify immediately!",
        "Congratulations! You've won $1,000,000. Claim your prize now!",
        "Your password will expire in 24 hours. Update it here: {link}",
        "Security Alert: Unusual activity detected. Verify your identity.",
        "Dear Customer, Your bank account needs verification. Act now!",
        "Limited time offer! Get 90% off. Click to claim your discount!",
        "Your package could not be delivered. Confirm your address here.",
        "IRS Notice: You have an outstanding tax refund. Claim it now!",
    ]
    
    # Normal email patterns
    normal_templates = [
        "Hi team, please find the meeting notes attached.",
        "Thank you for your order. Your shipment is on the way.",
        "Weekly newsletter: Check out our latest blog posts.",
        "Your monthly statement is now available in your account.",
        "Reminder: Team meeting tomorrow at 10 AM.",
        "Project update: We've completed phase 1 successfully.",
        "Happy birthday! Wishing you a wonderful day.",
        "Your subscription has been renewed for another year.",
    ]
    
    # Generate training data
    for i in range(2500):
        template = phishing_templates[i % len(phishing_templates)]
        texts.append(template + f" Reference: {i}")
        labels.append(1)
    
    for i in range(2500):
        template = normal_templates[i % len(normal_templates)]
        texts.append(template + f" ID: {i}")
        labels.append(0)
    
    # Load spam corpus if available
    spam_path = base_path / 'datasets/spam'
    if spam_path.exists():
        # Add any extracted spam data here
        pass
    
    return texts, np.array(labels)


def train_content_detector(X: np.ndarray, y: np.ndarray, save_path: str = None):
    """Train XGBoost classifier for content detection."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=4,
        random_state=42,
        tree_method='hist',
        scale_pos_weight=1,
        eval_metric='logloss'
    )
    
    print("Training XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Phishing/Spam']))
    
    if save_path:
        joblib.dump(model, save_path)
        print(f"\nModel saved to {save_path}")
    
    return model


if __name__ == '__main__':
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Load data
    print("Loading content data...")
    texts, y = load_content_data(base)
    print(f"Total texts: {len(texts)}")
    
    # Extract TF-IDF features
    print("Extracting TF-IDF features...")
    extractor = ContentFeatureExtractor(max_features=1000)
    X = extractor.fit_transform(texts)
    del texts
    gc.collect()
    
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Phishing/Spam ratio: {y.mean():.2%}")
    
    # Train
    model = train_content_detector(
        X, y,
        save_path=str(base / 'models/content_detector.pkl')
    )
    
    # Save vectorizer
    joblib.dump(extractor, base / 'models/content_vectorizer.pkl')
    print("Vectorizer saved")
