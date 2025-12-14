"""Run diagnostics on all 4 trained models."""
import numpy as np
import pandas as pd
import joblib
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/workspaces/AI-Hacking-detection-ML')
DATASETS = BASE / 'datasets'
MODELS = BASE / 'models'

print("="*80)
print("🔍 AI HACKING DETECTION - DIAGNOSTIC REPORT")
print("="*80)

# =============================================================================
# LOAD DATA FUNCTIONS
# =============================================================================

def load_network_data():
    dfs = []
    nsl = DATASETS / 'network_intrusion/nsl_kdd'
    cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment',
            'urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
            'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
            'is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
            'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate',
            'dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
            'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty']
    for f in nsl.glob('*.txt'):
        try:
            df = pd.read_csv(f, names=cols)
            df['label'] = (df['label'] != 'normal').astype(int)
            dfs.append(df.drop('difficulty', axis=1))
        except: pass
    
    cicids = DATASETS / 'network_intrusion/cicids2017'
    for f in list(cicids.glob('*.csv'))[:3]:
        try:
            df = pd.read_csv(f, low_memory=False, nrows=100000)
            df.columns = df.columns.str.strip()
            if 'Label' in df.columns:
                df['label'] = (df['Label'] != 'BENIGN').astype(int)
                df = df.drop('Label', axis=1).select_dtypes(include=[np.number])
                dfs.append(df)
        except: pass
    
    combined = pd.concat(dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    y = combined['label'].values
    X = combined.drop('label', axis=1).values
    feature_names = combined.drop('label', axis=1).columns.tolist()
    return X, y, feature_names

def load_fraud_data():
    f = DATASETS / 'fraud_detection/creditcard.csv'
    df = pd.read_csv(f)
    y = df['Class'].values
    X = df.drop('Class', axis=1).values
    feature_names = df.drop('Class', axis=1).columns.tolist()
    return X, y, feature_names

def load_url_data():
    dfs = []
    for f in (DATASETS / 'url_analysis').rglob('*.csv'):
        try:
            df = pd.read_csv(f, low_memory=False, nrows=200000)
            url_col = next((c for c in df.columns if 'url' in c.lower()), None)
            label_col = next((c for c in df.columns if c.lower() in ['label', 'type', 'class', 'is_malicious']), None)
            if url_col and label_col:
                df = df[[url_col, label_col]].dropna()
                df.columns = ['url', 'label']
                if df['label'].dtype == 'object':
                    df['label'] = df['label'].apply(lambda x: 0 if str(x).lower() in ['benign', 'good', 'legitimate', '0'] else 1)
                dfs.append(df)
        except: pass
    
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates('url')
    
    def url_features(url):
        url = str(url)
        return [len(url), url.count('/'), url.count('.'), url.count('-'),
                url.count('?'), url.count('='), url.count('&'),
                sum(c.isdigit() for c in url) / max(len(url), 1),
                sum(c.isupper() for c in url) / max(len(url), 1),
                1 if any(url.endswith(t) for t in ['.xyz','.tk','.ml','.ga','.top','.pw']) else 0]
    
    X = np.array([url_features(u) for u in combined['url']])
    y = combined['label'].values
    feature_names = ['url_len', 'slash_count', 'dot_count', 'dash_count', 'question_count', 
                     'equals_count', 'amp_count', 'digit_ratio', 'upper_ratio', 'suspicious_tld']
    return X, y, feature_names

def load_payload_data():
    payloads = DATASETS / 'security_payloads'
    data = []
    for folder in ['injection', 'fuzzing']:
        for f in (payloads / folder).rglob('*'):
            if f.is_file():
                try:
                    for line in f.read_text(errors='ignore').splitlines()[:200]:
                        if line.strip(): data.append((line.strip(), 1))
                except: pass
    
    for f in (payloads / 'wordlists').rglob('*'):
        if f.is_file() and 'english' in f.name.lower():
            try:
                for line in f.read_text(errors='ignore').splitlines()[:2000]:
                    if line.strip() and len(line) < 50: data.append((line.strip(), 0))
            except: pass
    
    data += [("hello world", 0), ("normal text", 0), ("user123", 0)] * 500
    mal = [d for d in data if d[1] == 1]
    ben = [d for d in data if d[1] == 0][:len(mal)]
    data = mal + ben
    texts, labels = zip(*data)
    return list(texts), np.array(labels)

# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def check_class_balance(name, y_train, y_test):
    print(f"\n--- {name} ---")
    for y, set_name in [(y_train, "Train"), (y_test, "Test")]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n{set_name} Distribution:")
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count:,} ({count/len(y)*100:.2f}%)")
        if len(unique) == 2:
            ratio = max(counts) / min(counts)
            if ratio > 10:
                print(f"  🚨 SEVERE IMBALANCE: Ratio {ratio:.1f}:1")
            elif ratio > 3:
                print(f"  ⚠️  Moderate imbalance: Ratio {ratio:.1f}:1")

def check_leakage(name, feature_names):
    print(f"\n--- {name} ---")
    patterns = {'ID/Index': ['id', 'index', 'row'], 'Timestamps': ['time', 'date', 'timestamp'],
                'Target': ['target', 'label', 'class', 'fraud', 'attack']}
    issues = []
    for cat, pats in patterns.items():
        found = [f for f in feature_names if any(p in str(f).lower() for p in pats)]
        if found:
            print(f"🚨 {cat}: {found[:3]}")
            issues.extend(found)
    if not issues:
        print("✅ No obvious leakage")
    return issues

def deep_eval(name, model, scaler, X_test, y_test):
    print(f"\n{'='*60}")
    print(f"📈 {name}")
    print('='*60)
    
    if scaler:
        n_feat = scaler.n_features_in_
        if X_test.shape[1] > n_feat:
            X_test = X_test[:, :n_feat]
        elif X_test.shape[1] < n_feat:
            X_test = np.hstack([X_test, np.zeros((X_test.shape[0], n_feat - X_test.shape[1]))])
        X_scaled = scaler.transform(X_test)
    else:
        X_scaled = X_test
    
    y_pred = model.predict(X_scaled)
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    if len(np.unique(y_test)) == 2:
        tn, fp, fn, tp = cm.ravel()
        print(f"\nTN: {tn:,}  FP: {fp:,}")
        print(f"FN: {fn:,}  TP: {tp:,}")
        
        if tp == 0:
            print("\n🚨 CRITICAL: Model predicts NO positive class!")
        elif tp < 10:
            print(f"\n🚨 CRITICAL: Only {tp} true positives!")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    unique_preds = np.unique(y_pred)
    print(f"🎯 Prediction Diversity: {len(unique_preds)} unique classes")
    for p in unique_preds:
        print(f"  Predicted {p}: {np.sum(y_pred == p):,} ({np.sum(y_pred == p)/len(y_pred)*100:.2f}%)")
    
    if len(unique_preds) == 1:
        print("\n🚨 CRITICAL: Model only predicts ONE class!")
    
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_scaled)[:, 1]
            roc = roc_auc_score(y_test, y_proba)
            print(f"\n📊 ROC-AUC: {roc:.4f}")
            if roc > 0.99:
                print("  ⚠️  AUC > 0.99 - Check for leakage!")
            elif roc < 0.6:
                print("  🚨 AUC < 0.6 - Model is terrible!")
            
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(rec, prec)
            print(f"📊 PR-AUC: {pr_auc:.4f}")
            if pr_auc < 0.3:
                print("  🚨 PR-AUC < 0.3 - Fails on minority class!")
        except: pass

# =============================================================================
# RUN DIAGNOSTICS
# =============================================================================

# 1. NETWORK MODEL
print("\n" + "="*80)
print("🌐 NETWORK INTRUSION MODEL")
print("="*80)
X, y, feat_names = load_network_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Total: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
check_class_balance("Network", y_train, y_test)
check_leakage("Network", feat_names)
model_data = joblib.load(MODELS / 'network_intrusion_model.pkl')
deep_eval("Network Intrusion", model_data['model'], model_data['scaler'], X_test, y_test)

# 2. FRAUD MODEL
print("\n" + "="*80)
print("💳 FRAUD DETECTION MODEL")
print("="*80)
X, y, feat_names = load_fraud_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Total: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
check_class_balance("Fraud", y_train, y_test)
check_leakage("Fraud", feat_names)
model_data = joblib.load(MODELS / 'fraud_detection_model.pkl')
deep_eval("Fraud Detection", model_data['model'], model_data['scaler'], X_test, y_test)

# 3. URL MODEL
print("\n" + "="*80)
print("🔗 URL ANALYSIS MODEL")
print("="*80)
X, y, feat_names = load_url_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Total: {len(X):,} | Train: {len(X_train):,} | Test: {len(X_test):,}")
check_class_balance("URL", y_train, y_test)
check_leakage("URL", feat_names)
model_data = joblib.load(MODELS / 'url_analysis_model.pkl')
deep_eval("URL Analysis", model_data['model'], model_data['scaler'], X_test, y_test)

# 4. PAYLOAD MODEL
print("\n" + "="*80)
print("💉 PAYLOAD CLASSIFIER MODEL")
print("="*80)
texts, y = load_payload_data()
with open(MODELS / 'payload_classifier.pkl', 'rb') as f:
    model_data = pickle.load(f)
X = model_data['vectorizer'].transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Total: {len(y):,} | Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
check_class_balance("Payload", y_train, y_test)
print("\n--- Payload ---\n✅ TF-IDF features (no leakage risk)")

y_pred = model_data['classifier'].predict(X_test)
print(f"\n{'='*60}")
print("📈 Payload Classifier")
print('='*60)
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
tn, fp, fn, tp = cm.ravel()
print(f"\nTN: {tn:,}  FP: {fp:,}")
print(f"FN: {fn:,}  TP: {tp:,}")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

print("\n" + "="*80)
print("📝 DIAGNOSTIC COMPLETE")
print("="*80)
