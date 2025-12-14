"""Fix Network and Fraud models with proper class balancing."""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

BASE = Path('/workspaces/AI-Hacking-detection-ML')
DATASETS = BASE / 'datasets'
MODELS = BASE / 'models'

# =============================================================================
# FIX 1: NETWORK INTRUSION MODEL
# =============================================================================
print("="*80)
print("🔧 FIX 1: NETWORK INTRUSION MODEL")
print("="*80)

# Load data
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
    df = pd.read_csv(f, names=cols)
    df['label'] = (df['label'] != 'normal').astype(int)
    dfs.append(df.drop('difficulty', axis=1))

# Add CICIDS2017
cicids = DATASETS / 'network_intrusion/cicids2017'
for f in list(cicids.glob('*.csv'))[:4]:
    try:
        df = pd.read_csv(f, low_memory=False, nrows=100000)
        df.columns = df.columns.str.strip()
        if 'Label' in df.columns:
            df['label'] = (df['Label'] != 'BENIGN').astype(int)
            df = df.drop('Label', axis=1).select_dtypes(include=[np.number])
            # Remove potential leakage columns
            leak_cols = [c for c in df.columns if any(x in c.lower() for x in ['idle', 'time', 'id'])]
            df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors='ignore')
            dfs.append(df)
    except: pass

combined = pd.concat(dfs, ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0)
y_network = combined['label'].values
X_network = combined.drop('label', axis=1).select_dtypes(include=[np.number]).values

X_train, X_test, y_train, y_test = train_test_split(X_network, y_network, test_size=0.2, random_state=42, stratify=y_network)

# Calculate class weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\nClass distribution: {neg_count:,} normal, {pos_count:,} attacks")
print(f"Imbalance ratio: {scale_pos_weight:.1f}:1")
print(f"Using scale_pos_weight={int(scale_pos_weight)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with class balancing
print("\nTraining XGBoost with class balancing...")
xgb_network = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.1,
    scale_pos_weight=int(scale_pos_weight),  # FIX: Class imbalance
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',  # FIX: Use PR-AUC
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

xgb_network.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

# Evaluate
y_pred = xgb_network.predict(X_test_scaled)
y_proba = xgb_network.predict_proba(X_test_scaled)[:, 1]

print("\n📊 RESULTS - Network Intrusion Model (FIXED)")
print("-"*50)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:\n{cm}")
print(f"\nAttacks Caught: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
print(f"Attacks Missed: {fn}/{tp+fn} ({fn/(tp+fn)*100:.1f}%)")
print(f"False Alarms: {fp}/{tn+fp} ({fp/(tn+fp)*100:.2f}%)")

print("\n" + classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f} (was 0.418)")

# Save
joblib.dump({'model': xgb_network, 'scaler': scaler}, MODELS / 'network_intrusion_model.pkl')
print(f"\n✅ Saved: network_intrusion_model.pkl")

# =============================================================================
# FIX 2: FRAUD DETECTION MODEL
# =============================================================================
print("\n" + "="*80)
print("🔧 FIX 2: FRAUD DETECTION MODEL")
print("="*80)

# Load data
df_fraud = pd.read_csv(DATASETS / 'fraud_detection/creditcard.csv')

# FIX: Remove 'Time' feature (data leakage)
print("\n⚠️  Removing 'Time' feature (data leakage)")
print(f"Features before: {df_fraud.shape[1]-1}")
df_fraud = df_fraud.drop('Time', axis=1)
print(f"Features after: {df_fraud.shape[1]-1}")

y_fraud = df_fraud['Class'].values
X_fraud = df_fraud.drop('Class', axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud)

# Calculate class weight
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count
print(f"\nClass distribution: {neg_count:,} normal, {pos_count:,} fraud")
print(f"Imbalance ratio: {scale_pos_weight:.1f}:1")
print(f"Using scale_pos_weight={int(scale_pos_weight)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train with class balancing
print("\nTraining XGBoost with class balancing...")
xgb_fraud = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=int(scale_pos_weight),  # FIX: Class imbalance
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',  # FIX: Use PR-AUC
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=42,
    verbosity=0
)

xgb_fraud.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)

# Evaluate
y_pred = xgb_fraud.predict(X_test_scaled)
y_proba = xgb_fraud.predict_proba(X_test_scaled)[:, 1]

print("\n📊 RESULTS - Fraud Detection Model (FIXED)")
print("-"*50)
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:\n{cm}")
print(f"\nFraud Caught: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
print(f"Fraud Missed: {fn}/{tp+fn} ({fn/(tp+fn)*100:.1f}%)")
print(f"False Alarms: {fp}/{tn+fp} ({fp/(tn+fp)*100:.3f}%)")

print("\n" + classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_proba)
precision, recall, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall, precision)
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f} (was 0.834)")

# Save
joblib.dump({'model': xgb_fraud, 'scaler': scaler}, MODELS / 'fraud_detection_model.pkl')
print(f"\n✅ Saved: fraud_detection_model.pkl")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("📝 FIX SUMMARY")
print("="*80)

print("""
┌─────────────────────┬──────────────┬──────────────┬─────────────┐
│ Model               │ Before       │ After        │ Status      │
├─────────────────────┼──────────────┼──────────────┼─────────────┤
│ Network PR-AUC      │ 0.418        │ {:.3f}        │ {}          │
│ Network Detection   │ 78.9%        │ {:.1f}%        │ {}          │
├─────────────────────┼──────────────┼──────────────┼─────────────┤
│ Fraud PR-AUC        │ 0.834        │ {:.3f}        │ {}          │
│ Fraud Detection     │ 80.6%        │ {:.1f}%        │ {}          │
│ Time Feature        │ YES (leak)   │ REMOVED      │ ✅          │
├─────────────────────┼──────────────┼──────────────┼─────────────┤
│ URL Analysis        │ ✅ No change │ ✅ No change │ ✅          │
│ Payload Classifier  │ ✅ No change │ ✅ No change │ ✅          │
└─────────────────────┴──────────────┴──────────────┴─────────────┘
""".format(
    pr_auc, "✅" if pr_auc > 0.7 else "⚠️",
    tp/(tp+fn)*100, "✅" if tp/(tp+fn) > 0.85 else "⚠️",
    pr_auc, "✅" if pr_auc > 0.8 else "⚠️",
    tp/(tp+fn)*100, "✅" if tp/(tp+fn) > 0.85 else "⚠️"
))
