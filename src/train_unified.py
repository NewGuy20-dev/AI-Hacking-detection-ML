"""Train unified Network Detector with aligned features from NSL-KDD + CICIDS2017."""
import gc
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Common features that exist in both datasets (semantic mapping)
UNIFIED_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count',
    'serror_rate', 'rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_serror_rate', 'dst_host_rerror_rate'
]


def load_nsl_unified(path: str):
    """Load NSL-KDD with unified features."""
    cols = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty']
    
    df = pd.read_csv(path, names=cols)
    df['is_attack'] = (df['label'] != 'normal').astype(int)
    return df[UNIFIED_FEATURES + ['is_attack']]


def load_cicids_unified(data_dir: str, sample_per_file: int = 15000):
    """Load CICIDS2017 with unified features mapped."""
    # Map CICIDS columns to unified names
    col_map = {
        'Flow Duration': 'duration',
        'Total Fwd Packets': 'src_bytes',  # proxy
        'Total Backward Packets': 'dst_bytes',  # proxy
        'Flow Packets/s': 'count',
        'Flow Bytes/s': 'srv_count',
        'Fwd PSH Flags': 'serror_rate',
        'Bwd PSH Flags': 'rerror_rate',
        'Fwd Packets/s': 'same_srv_rate',
        'Bwd Packets/s': 'diff_srv_rate',
        'Subflow Fwd Packets': 'dst_host_count',
        'Subflow Bwd Packets': 'dst_host_srv_count',
        'Init_Win_bytes_forward': 'dst_host_same_srv_rate',
        'Init_Win_bytes_backward': 'dst_host_diff_srv_rate',
        'Active Mean': 'dst_host_serror_rate',
        'Idle Mean': 'dst_host_rerror_rate',
    }
    
    dfs = []
    for csv_file in Path(data_dir).glob('*.csv'):
        print(f"  Loading {csv_file.name}...")
        df = pd.read_csv(csv_file, low_memory=False)
        df.columns = df.columns.str.strip()
        
        if len(df) > sample_per_file:
            df = df.sample(n=sample_per_file, random_state=42)
        
        # Map columns
        mapped = pd.DataFrame()
        for cicids_col, unified_col in col_map.items():
            if cicids_col in df.columns:
                mapped[unified_col] = pd.to_numeric(df[cicids_col], errors='coerce')
        
        mapped['is_attack'] = (df['Label'] != 'BENIGN').astype(int)
        dfs.append(mapped)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    return combined


def train_unified():
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Load both datasets with unified features
    print("Loading NSL-KDD (unified)...")
    df_nsl_train = load_nsl_unified(str(base / 'data/nsl_kdd_train.csv'))
    df_nsl_test = load_nsl_unified(str(base / 'data/nsl_kdd_test.csv'))
    df_nsl = pd.concat([df_nsl_train, df_nsl_test], ignore_index=True)
    print(f"  NSL-KDD: {len(df_nsl)} samples")
    
    print("\nLoading CICIDS2017 (unified)...")
    df_cicids = load_cicids_unified(str(base / 'cicids2017/MachineLearningCVE'), sample_per_file=15000)
    print(f"  CICIDS2017: {len(df_cicids)} samples")
    
    # Combine
    df_all = pd.concat([df_nsl, df_cicids], ignore_index=True)
    del df_nsl, df_cicids
    gc.collect()
    
    X = df_all[UNIFIED_FEATURES].astype('float32').values
    y = df_all['is_attack'].values
    
    # Scale
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    print(f"\nCombined: {len(y)} samples, {len(UNIFIED_FEATURES)} features")
    print(f"Attack ratio: {y.mean():.2%}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    model = RandomForestClassifier(
        n_estimators=100, max_depth=20, min_samples_split=5,
        n_jobs=4, random_state=42, class_weight='balanced'
    )
    
    print("\nTraining unified model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    # Save
    joblib.dump(model, base / 'models/network_detector.pkl')
    joblib.dump(scaler, base / 'models/network_scaler.pkl')
    joblib.dump(UNIFIED_FEATURES, base / 'models/feature_names.pkl')
    print("Model saved!")


if __name__ == '__main__':
    train_unified()
