"""Train 3 specialized models: Network Intrusion, Fraud Detection, URL Analysis."""
import gc
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                             recall_score, average_precision_score, classification_report)
import xgboost as xgb
import lightgbm as lgb

BASE = Path('/workspaces/AI-Hacking-detection-ML')


def train_network_model():
    """Train Network Intrusion Detection model."""
    print("\n" + "="*60)
    print("TRAINING: Network Intrusion Detection Model")
    print("="*60)
    
    dfs = []
    
    # NSL-KDD
    print("Loading NSL-KDD...")
    cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
            'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
            'root_shell','su_attempted','num_root','num_file_creations','num_shells',
            'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
            'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
            'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
            'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
            'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']
    
    df = pd.read_csv(BASE / 'data/nsl_kdd_train.csv', names=cols)
    df['is_attack'] = (df['label'] != 'normal').astype(int)
    dfs.append(df[['duration','src_bytes','dst_bytes','count','srv_count','serror_rate',
                   'same_srv_rate','dst_host_count','dst_host_srv_count','is_attack']])
    
    # CICIDS2017 (sample)
    print("Loading CICIDS2017...")
    for f in list((BASE / 'cicids2017/MachineLearningCVE').glob('*.csv'))[:4]:
        df = pd.read_csv(f, low_memory=False, nrows=20000)
        df.columns = df.columns.str.strip()
        df['is_attack'] = (df['Label'] != 'BENIGN').astype(int)
        df = df[['Flow Duration','Total Fwd Packets','Total Backward Packets',
                 'Flow Packets/s','Flow Bytes/s','Fwd Packets/s','Bwd Packets/s',
                 'Subflow Fwd Packets','Subflow Bwd Packets','is_attack']].copy()
        df.columns = ['duration','src_bytes','dst_bytes','count','srv_count','serror_rate',
                      'same_srv_rate','dst_host_count','dst_host_srv_count','is_attack']
        dfs.append(df)
    
    # UNSW-NB15 (sample)
    print("Loading UNSW-NB15...")
    df = pd.read_csv(BASE / 'datasets/unsw_nb15/UNSW_NB15_training-set.csv', nrows=50000)
    df['is_attack'] = df['label'].astype(int)
    df = df[['dur','sbytes','dbytes','sload','dload','spkts','dpkts','stcpb','dtcpb','is_attack']].copy()
    df.columns = ['duration','src_bytes','dst_bytes','count','srv_count','serror_rate',
                  'same_srv_rate','dst_host_count','dst_host_srv_count','is_attack']
    dfs.append(df)
    
    # Cyber Attacks
    print("Loading Cyber Attacks...")
    df = pd.read_csv(BASE / 'datasets/cyber_attacks/cybersecurity_attacks.csv')
    df['is_attack'] = (df['Attack Type'] != 'Normal').astype(int) if 'Attack Type' in df.columns else 1
    df = df[['Source Port','Destination Port','Packet Length','Anomaly Scores','is_attack']].copy()
    df.columns = ['duration','src_bytes','dst_bytes','count','is_attack']
    df['srv_count'] = df['count']
    df['serror_rate'] = 0
    df['same_srv_rate'] = 0
    df['dst_host_count'] = df['src_bytes']
    df['dst_host_srv_count'] = df['dst_bytes']
    dfs.append(df[['duration','src_bytes','dst_bytes','count','srv_count','serror_rate',
                   'same_srv_rate','dst_host_count','dst_host_srv_count','is_attack']])
    
    # Combine
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    del dfs
    gc.collect()
    
    X = combined.drop('is_attack', axis=1).astype('float32').values
    y = combined['is_attack'].values
    print(f"Total samples: {len(y)}, Attack ratio: {y.mean():.2%}")
    
    # Scale
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=4, random_state=42, class_weight='balanced')
    print("Training RandomForest...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    print(f"\nMetrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    joblib.dump({'model': model, 'scaler': scaler}, BASE / 'models/network_intrusion_model.pkl')
    return metrics


def train_fraud_model():
    """Train Fraud Detection model."""
    print("\n" + "="*60)
    print("TRAINING: Fraud Detection Model")
    print("="*60)
    
    print("Loading Credit Card Fraud dataset...")
    df = pd.read_csv(BASE / 'datasets/fraud/creditcard.csv')
    
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    print(f"Total samples: {len(y)}, Fraud ratio: {y.mean():.4%}")
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # XGBoost handles imbalanced data well
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=100,
                              n_jobs=4, random_state=42, eval_metric='aucpr')
    print("Training XGBoost...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc_pr': average_precision_score(y_test, y_prob)
    }
    print(f"\nMetrics: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, AUC-PR={metrics['auc_pr']:.4f}")
    
    joblib.dump({'model': model, 'scaler': scaler}, BASE / 'models/fraud_detection_model.pkl')
    return metrics


def train_url_model():
    """Train URL Analysis model."""
    print("\n" + "="*60)
    print("TRAINING: URL Analysis Model")
    print("="*60)
    
    import math
    from urllib.parse import urlparse
    
    def extract_features(url):
        try:
            parsed = urlparse(url if '://' in str(url) else f'http://{url}')
            domain = parsed.netloc or str(url).split('/')[0]
        except:
            domain = str(url)
        
        url_str = str(url)
        entropy = 0
        if url_str:
            prob = [url_str.count(c)/len(url_str) for c in set(url_str)]
            entropy = -sum(p * math.log2(p) for p in prob if p > 0)
        
        return [
            len(url_str),
            len(domain),
            url_str.count('/'),
            url_str.count('.'),
            url_str.count('-'),
            sum(c.isdigit() for c in url_str) / max(len(url_str), 1),
            entropy,
            1 if any(url_str.endswith(t) for t in ['.xyz','.tk','.ml','.ga','.top']) else 0
        ]
    
    print("Loading Malicious URLs dataset...")
    df = pd.read_csv(BASE / 'datasets/malware_urls/malicious_phish.csv')
    
    # Map labels
    label_map = {'benign': 0, 'defacement': 1, 'phishing': 1, 'malware': 1}
    df['is_malicious'] = df['type'].map(label_map).fillna(1).astype(int)
    
    print("Extracting URL features...")
    features = [extract_features(url) for url in df['url'].values]
    X = np.array(features, dtype='float32')
    y = df['is_malicious'].values
    
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    print(f"Total samples: {len(y)}, Malicious ratio: {y.mean():.2%}")
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = lgb.LGBMClassifier(n_estimators=100, max_depth=15, n_jobs=4, random_state=42, verbose=-1)
    print("Training LightGBM...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    print(f"\nMetrics: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    joblib.dump({'model': model, 'scaler': scaler}, BASE / 'models/url_analysis_model.pkl')
    return metrics


if __name__ == '__main__':
    all_metrics = {}
    all_metrics['network'] = train_network_model()
    all_metrics['fraud'] = train_fraud_model()
    all_metrics['url'] = train_url_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE - ALL METRICS")
    print("="*60)
    for model_name, metrics in all_metrics.items():
        print(f"\n{model_name.upper()}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    joblib.dump(all_metrics, BASE / 'models/training_metrics.pkl')
