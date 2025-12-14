"""Data loading and preprocessing for AI Hacking Detection."""
import pandas as pd
import numpy as np
import gc
from pathlib import Path

# NSL-KDD column names
NSL_KDD_COLS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

ATTACK_TYPES = {
    'normal': 'normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS', 'smurf': 'DoS', 'teardrop': 'DoS',
    'mailbomb': 'DoS', 'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
    'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe', 'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L', 'multihop': 'R2L', 'phf': 'R2L',
    'spy': 'R2L', 'warezclient': 'R2L', 'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'worm': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R', 'rootkit': 'U2R',
    'httptunnel': 'U2R', 'ps': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R'
}


def load_nsl_kdd(train_path: str, test_path: str = None, sample_size: int = None):
    """Load NSL-KDD dataset."""
    df = pd.read_csv(train_path, names=NSL_KDD_COLS)
    
    if test_path:
        df_test = pd.read_csv(test_path, names=NSL_KDD_COLS)
        df = pd.concat([df, df_test], ignore_index=True)
        del df_test
        gc.collect()
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    # Map attack types
    df['attack_type'] = df['label'].map(lambda x: ATTACK_TYPES.get(x, 'unknown'))
    df['is_attack'] = (df['label'] != 'normal').astype(int)
    
    return df


def load_phishing_arff(path: str):
    """Load phishing dataset from ARFF format."""
    data = []
    with open(path, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.lower() == '@data':
                in_data = True
                continue
            if in_data and line and not line.startswith('%'):
                data.append(line.split(','))
    
    df = pd.DataFrame(data)
    df.columns = [f'f{i}' for i in range(len(df.columns) - 1)] + ['label']
    df['is_attack'] = (df['label'] == '1').astype(int)
    return df


def preprocess_network_data(df: pd.DataFrame, for_prediction: bool = False):
    """Preprocess network data for training or prediction."""
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    
    df = df.copy()
    cat_cols = ['protocol_type', 'service', 'flag']
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Drop non-feature columns
    drop_cols = ['label', 'attack_type', 'difficulty', 'is_attack']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].astype('float32')
    
    y = None
    if 'is_attack' in df.columns:
        y = df['is_attack'].values
    elif 'label' in df.columns and not for_prediction:
        y = (df['label'] != 'normal').astype(int).values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols


def load_cicids2017(data_dir: str, sample_per_file: int = 15000):
    """Load CICIDS2017 dataset with sampling for memory efficiency."""
    from pathlib import Path
    
    data_path = Path(data_dir)
    dfs = []
    
    for csv_file in data_path.glob('*.csv'):
        print(f"  Loading {csv_file.name}...")
        df = pd.read_csv(csv_file, low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Sample if too large
        if len(df) > sample_per_file:
            df = df.sample(n=sample_per_file, random_state=42)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    
    # Create binary label
    combined['is_attack'] = (combined['Label'] != 'BENIGN').astype(int)
    
    # Clean infinite/nan values
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.fillna(0)
    
    return combined


def preprocess_cicids_data(df: pd.DataFrame):
    """Preprocess CICIDS2017 data."""
    from sklearn.preprocessing import MinMaxScaler
    
    df = df.copy()
    drop_cols = ['Label', 'is_attack']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Convert to numeric
    X = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32')
    y = df['is_attack'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_cols


if __name__ == '__main__':
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Load NSL-KDD
    print("Loading NSL-KDD...")
    df = load_nsl_kdd(
        str(base / 'data/nsl_kdd_train.csv'),
        str(base / 'data/nsl_kdd_test.csv')
    )
    print(f"Loaded {len(df)} records")
    print(f"Attack distribution:\n{df['is_attack'].value_counts()}")
    
    X, y, scaler, cols = preprocess_network_data(df)
    print(f"Features shape: {X.shape}")
    
    # Save processed data
    np.save(base / 'data/processed/network_X.npy', X)
    np.save(base / 'data/processed/network_y.npy', y)
    print("Saved processed network data")
