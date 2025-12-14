"""Validate unified network detector."""
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_auc_score)

from train_unified import load_nsl_unified, load_cicids_unified, UNIFIED_FEATURES


def validate():
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    model = joblib.load(base / 'models/network_detector.pkl')
    scaler = joblib.load(base / 'models/network_scaler.pkl')
    
    print("=" * 60)
    print("UNIFIED MODEL VALIDATION")
    print("=" * 60)
    
    # Test on NSL-KDD
    print("\n[1] NSL-KDD Test Set")
    print("-" * 40)
    df = load_nsl_unified(str(base / 'data/nsl_kdd_test.csv'))
    X = scaler.transform(df[UNIFIED_FEATURES].astype('float32'))
    y = df['is_attack'].values
    
    y_pred = model.predict(X)
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y, model.predict_proba(X)[:, 1]):.4f}")
    print(f"\n{confusion_matrix(y, y_pred)}")
    
    # Test on CICIDS2017
    print("\n[2] CICIDS2017 Test Set")
    print("-" * 40)
    df = load_cicids_unified(str(base / 'cicids2017/MachineLearningCVE'), sample_per_file=5000)
    X = scaler.transform(df[UNIFIED_FEATURES].astype('float32'))
    y = df['is_attack'].values
    
    y_pred = model.predict(X)
    print(f"Accuracy:  {accuracy_score(y, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y, model.predict_proba(X)[:, 1]):.4f}")
    print(f"\n{confusion_matrix(y, y_pred)}")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    validate()
