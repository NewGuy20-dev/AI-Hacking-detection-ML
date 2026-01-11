#!/usr/bin/env python3
"""Train Host Behavior model."""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def main():
    base = Path(__file__).parent.parent
    synth_path = base / "datasets/host_behavior/synthetic_500k.jsonl"
    live_benign_path = base / "datasets/live_benign/host_behavior_benign.jsonl"
    
    print("Loading data...")
    samples = []
    
    # Load malicious (synthetic)
    if synth_path.exists():
        samples.extend([json.loads(l) for l in tqdm(open(synth_path), desc="Loading synthetic")])
    
    # Load benign (live_benign)
    if live_benign_path.exists():
        samples.extend([json.loads(l) for l in tqdm(open(live_benign_path), desc="Loading live benign")])
    
    df = pd.DataFrame(samples)
    print(f"Samples: {len(df):,}, Malware: {df['label'].sum():,}, Benign: {(df['label']==0).sum():,}")
    
    features = [c for c in df.columns if c not in ['label', 'category']]
    X = df[features].values.astype(np.float32)
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    print("Training RandomForest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    joblib.dump(model, base / "models/host_behavior_model.pkl")
    joblib.dump(scaler, base / "models/host_behavior_scaler.pkl")
    print("✓ Saved host_behavior_model.pkl")

if __name__ == "__main__":
    main()
