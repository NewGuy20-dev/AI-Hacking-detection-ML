#!/usr/bin/env python3
"""Train Network Intrusion model."""
import json, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

FEATURES = ['duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

def main():
    base = Path(__file__).parent.parent
    synth_path = base / "datasets/network_intrusion/synthetic_500k.jsonl"
    live_benign_path = base / "datasets/live_benign/mawi_network_kdd.jsonl"
    
    print("Loading data...")
    samples = []
    
    # Load malicious (synthetic)
    if synth_path.exists():
        samples.extend([json.loads(l) for l in tqdm(open(synth_path), desc="Loading synthetic")])
    
    # Load benign (MAWI live network traces)
    if live_benign_path.exists():
        samples.extend([json.loads(l) for l in tqdm(open(live_benign_path), desc="Loading MAWI benign")])
    
    df = pd.DataFrame(samples)
    print(f"Samples: {len(df):,}, Attack: {df['label'].sum():,}, Normal: {(df['label']==0).sum():,}")
    
    X = df[[f for f in FEATURES if f in df.columns]].fillna(0).values.astype(np.float32)
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    
    print("Training RandomForest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}")
    joblib.dump(model, base / "models/network_intrusion_model.pkl")
    joblib.dump(scaler, base / "models/network_scaler.pkl")
    print("✓ Saved network_intrusion_model.pkl")

if __name__ == "__main__":
    main()
