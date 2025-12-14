"""Train all 4 specialized models with organized datasets."""
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE = Path('/workspaces/AI-Hacking-detection-ML')
DATASETS = BASE / 'datasets'
MODELS = BASE / 'models'

def load_network_data():
    """Load all network intrusion datasets."""
    dfs = []
    
    # NSL-KDD
    nsl = DATASETS / 'network_intrusion/nsl_kdd'
    cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
            'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
            'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
            'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
            'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
            'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']
    for f in ['KDDTrain+.txt', 'KDDTest+.txt']:
        p = nsl / f
        if p.exists():
            df = pd.read_csv(p, names=cols)
            df['label'] = (df['label'] != 'normal').astype(int)
            dfs.append(df.drop('difficulty', axis=1))
    
    # CICIDS2017
    cicids = DATASETS / 'network_intrusion/cicids2017'
    for f in cicids.glob('*.csv'):
        try:
            df = pd.read_csv(f, low_memory=False)
            df.columns = df.columns.str.strip()
            if 'Label' in df.columns:
                df['label'] = (df['Label'] != 'BENIGN').astype(int)
                df = df.drop('Label', axis=1)
                df = df.select_dtypes(include=[np.number])
                df['label'] = df.get('label', 0)
                dfs.append(df.head(50000))  # Sample per file
        except: pass
    
    # UNSW-NB15
    unsw = DATASETS / 'network_intrusion/unsw_nb15'
    for f in unsw.glob('*.csv'):
        try:
            df = pd.read_csv(f, low_memory=False)
            if 'label' in df.columns:
                df = df.select_dtypes(include=[np.number])
                dfs.append(df.head(50000))
        except: pass
    
    if not dfs:
        return None, None
    
    # Combine and align features
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    y = combined['label'].values
    X = combined.drop('label', axis=1).select_dtypes(include=[np.number]).values
    
    print(f"Network data: {len(X)} samples, {X.shape[1]} features")
    return X, y

def load_fraud_data():
    """Load fraud detection data."""
    f = DATASETS / 'fraud_detection/creditcard.csv'
    if not f.exists():
        return None, None
    df = pd.read_csv(f)
    y = df['Class'].values
    X = df.drop('Class', axis=1).values
    print(f"Fraud data: {len(X)} samples")
    return X, y

def load_url_data():
    """Load URL analysis data."""
    dfs = []
    url_dir = DATASETS / 'url_analysis'
    
    for f in url_dir.rglob('*.csv'):
        try:
            df = pd.read_csv(f, low_memory=False)
            # Look for URL and label columns
            url_col = next((c for c in df.columns if 'url' in c.lower()), None)
            label_col = next((c for c in df.columns if c.lower() in ['label', 'type', 'class', 'is_malicious']), None)
            if url_col and label_col:
                df = df[[url_col, label_col]].dropna()
                df.columns = ['url', 'label']
                # Convert labels to binary
                if df['label'].dtype == 'object':
                    df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['benign', 'good', 'legitimate', '0'] else 1)
                dfs.append(df)
        except: pass
    
    if not dfs:
        return None, None
    
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates('url')
    
    # Extract URL features
    def url_features(url):
        url = str(url)
        return [
            len(url), url.count('/'), url.count('.'), url.count('-'),
            url.count('?'), url.count('='), url.count('&'),
            sum(c.isdigit() for c in url) / max(len(url), 1),
            sum(c.isupper() for c in url) / max(len(url), 1),
            1 if any(url.endswith(t) for t in ['.xyz','.tk','.ml','.ga','.top','.pw']) else 0
        ]
    
    X = np.array([url_features(u) for u in combined['url']])
    y = combined['label'].values
    print(f"URL data: {len(X)} samples")
    return X, y

def load_payload_data():
    """Load security payload data."""
    payloads = DATASETS / 'security_payloads'
    data = []
    
    # Malicious payloads
    for folder in ['injection', 'fuzzing']:
        for f in (payloads / folder).rglob('*'):
            if f.is_file() and f.suffix in ('', '.txt', '.lst', '.list'):
                try:
                    for line in f.read_text(errors='ignore').splitlines()[:500]:
                        if line.strip():
                            data.append((line.strip(), 1))
                except: pass
    
    mal_count = len(data)
    
    # Benign - common words
    for f in (payloads / 'wordlists').rglob('*'):
        if f.is_file() and 'english' in f.name.lower():
            try:
                for line in f.read_text(errors='ignore').splitlines()[:5000]:
                    if line.strip() and len(line) < 50 and line.isalpha():
                        data.append((line.strip(), 0))
            except: pass
    
    # Add synthetic benign
    benign = ["hello world", "user123", "test@email.com", "john doe", "2024-01-01",
              "normal text", "username", "password123", "example.com", "search query"]
    data += [(b, 0) for b in benign] * 500
    
    # Balance
    mal = [d for d in data if d[1] == 1]
    ben = [d for d in data if d[1] == 0][:len(mal)]
    data = mal + ben
    
    if not data:
        return None, None, None
    
    texts, labels = zip(*data)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,3), analyzer='char_wb')
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)
    print(f"Payload data: {len(y)} samples")
    return X, y, vectorizer

def train_model(name, X, y, model_class, **kwargs):
    """Train and save a model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale if dense
    scaler = None
    if hasattr(X_train, 'toarray'):
        X_train_scaled, X_test_scaled = X_train, X_test
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    model = model_class(**kwargs)
    model.fit(X_train_scaled, y_train)
    
    acc = model.score(X_test_scaled, y_test)
    print(f"{name}: Accuracy={acc:.2%}")
    
    return {'model': model, 'scaler': scaler, 'accuracy': acc}

def main():
    MODELS.mkdir(exist_ok=True)
    results = {}
    
    # 1. Network Intrusion
    print("\n=== Training Network Intrusion Model ===")
    X, y = load_network_data()
    if X is not None:
        r = train_model('Network', X, y, RandomForestClassifier, n_estimators=100, n_jobs=-1, random_state=42)
        joblib.dump(r, MODELS / 'network_intrusion_model.pkl')
        results['network'] = r['accuracy']
    
    # 2. Fraud Detection
    print("\n=== Training Fraud Detection Model ===")
    X, y = load_fraud_data()
    if X is not None:
        r = train_model('Fraud', X, y, XGBClassifier, n_estimators=100, max_depth=6, random_state=42, eval_metric='logloss')
        joblib.dump(r, MODELS / 'fraud_detection_model.pkl')
        results['fraud'] = r['accuracy']
    
    # 3. URL Analysis
    print("\n=== Training URL Analysis Model ===")
    X, y = load_url_data()
    if X is not None:
        r = train_model('URL', X, y, LGBMClassifier, n_estimators=100, random_state=42, verbose=-1)
        joblib.dump(r, MODELS / 'url_analysis_model.pkl')
        results['url'] = r['accuracy']
    
    # 4. Payload Classifier
    print("\n=== Training Payload Classifier ===")
    X, y, vectorizer = load_payload_data()
    if X is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        print(f"Payload: Accuracy={acc:.2%}")
        with open(MODELS / 'payload_classifier.pkl', 'wb') as f:
            pickle.dump({'vectorizer': vectorizer, 'classifier': clf}, f)
        results['payload'] = acc
    
    print("\n=== Training Complete ===")
    for name, acc in results.items():
        print(f"  {name}: {acc:.2%}")
    
    joblib.dump(results, MODELS / 'training_metrics.pkl')

if __name__ == "__main__":
    main()
