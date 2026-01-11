#!/usr/bin/env python3
"""Train Fraud Detection model."""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def main():
    base = Path(__file__).parent.parent
    synth_path = base / "datasets/fraud_detection/synthetic_500k.jsonl"
    live_benign_path = base / "datasets/live_benign/fraud_benign.jsonl"
    
    print("Loading data...")
    samples = []
    
    # Load malicious (synthetic)
    if synth_path.exists():
        samples.extend([json.loads(l) for l in tqdm(open(synth_path), desc="Loading synthetic")])
    
    # Load benign (live_benign)
    if live_benign_path.exists():
        samples.extend([json.loads(l) for l in tqdm(open(live_benign_path), desc="Loading live benign")])
    
    df = pd.DataFrame(samples)
    print(f"Samples: {len(df):,}, Fraud: {df['Class'].sum():,}, Normal: {(df['Class']==0).sum():,}")
    
    features = [c for c in df.columns if c not in ['Class']]
    X = df[features].values.astype(np.float32)
    y = df['Class'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    print("Training XGBoost...")
    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    joblib.dump(model, base / "models/fraud_detection_model.pkl")
    joblib.dump(scaler, base / "models/fraud_scaler.pkl")
    print("✓ Saved fraud_detection_model.pkl")

if __name__ == "__main__":
    main()
