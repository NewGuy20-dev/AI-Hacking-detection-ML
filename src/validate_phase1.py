"""Validate Phase 1: Compare baseline vs enhanced models."""
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from train_unified import load_nsl_unified, load_cicids_unified, UNIFIED_FEATURES
from anomaly_detector import AnomalyDetector
from ensemble_voting import EnsembleVoting


def validate_phase1():
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    # Load test data
    print("Loading test data...")
    df_nsl = load_nsl_unified(str(base / 'data/nsl_kdd_test.csv'))
    df_cicids = load_cicids_unified(str(base / 'cicids2017/MachineLearningCVE'), sample_per_file=5000)
    
    X_nsl = df_nsl[UNIFIED_FEATURES].astype('float32').values
    y_nsl = df_nsl['is_attack'].values
    X_cicids = df_cicids[UNIFIED_FEATURES].astype('float32').values
    y_cicids = df_cicids['is_attack'].values
    
    results = {}
    
    # Baseline (single RF)
    print("\n[Baseline] Single RandomForest:")
    baseline = joblib.load(base / 'models/network_detector.pkl')
    scaler = joblib.load(base / 'models/network_scaler.pkl')
    
    for name, X, y in [('NSL-KDD', X_nsl, y_nsl), ('CICIDS', X_cicids, y_cicids)]:
        X_s = scaler.transform(X)
        pred = baseline.predict(X_s)
        results[f'baseline_{name}'] = {
            'accuracy': accuracy_score(y, pred),
            'f1': f1_score(y, pred)
        }
        print(f"  {name}: Acc={results[f'baseline_{name}']['accuracy']:.4f}, F1={results[f'baseline_{name}']['f1']:.4f}")
    
    # Ensemble Voting
    print("\n[Enhanced] Ensemble Voting (RF+XGB+LGB):")
    ensemble = EnsembleVoting.load(str(base / 'models/ensemble_voting.pkl'))
    
    for name, X, y in [('NSL-KDD', X_nsl, y_nsl), ('CICIDS', X_cicids, y_cicids)]:
        pred = ensemble.predict(X)
        results[f'ensemble_{name}'] = {
            'accuracy': accuracy_score(y, pred),
            'f1': f1_score(y, pred)
        }
        print(f"  {name}: Acc={results[f'ensemble_{name}']['accuracy']:.4f}, F1={results[f'ensemble_{name}']['f1']:.4f}")
    
    # Anomaly Detector
    print("\n[Enhanced] Anomaly Detector (zero-day):")
    anomaly = AnomalyDetector.load(str(base / 'models/anomaly_detector.pkl'))
    
    for name, X, y in [('NSL-KDD', X_nsl, y_nsl), ('CICIDS', X_cicids, y_cicids)]:
        result = anomaly.predict(X)
        pred = result['is_anomaly']
        results[f'anomaly_{name}'] = {
            'accuracy': accuracy_score(y, pred),
            'f1': f1_score(y, pred)
        }
        print(f"  {name}: Acc={results[f'anomaly_{name}']['accuracy']:.4f}, F1={results[f'anomaly_{name}']['f1']:.4f}")
    
    # Combined (ensemble + anomaly boost)
    print("\n[Combined] Ensemble + Anomaly boost:")
    for name, X, y in [('NSL-KDD', X_nsl, y_nsl), ('CICIDS', X_cicids, y_cicids)]:
        ens_proba = ensemble.predict_proba(X)[:, 1]
        anom_score = anomaly.predict(X)['anomaly_score']
        anom_norm = (anom_score - anom_score.min()) / (anom_score.max() - anom_score.min() + 1e-8)
        combined = 0.8 * ens_proba + 0.2 * anom_norm
        pred = (combined >= 0.5).astype(int)
        results[f'combined_{name}'] = {
            'accuracy': accuracy_score(y, pred),
            'f1': f1_score(y, pred)
        }
        print(f"  {name}: Acc={results[f'combined_{name}']['accuracy']:.4f}, F1={results[f'combined_{name}']['f1']:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 1 SUMMARY")
    print("="*60)
    print(f"Baseline F1 (avg): {(results['baseline_NSL-KDD']['f1'] + results['baseline_CICIDS']['f1'])/2:.4f}")
    print(f"Ensemble F1 (avg): {(results['ensemble_NSL-KDD']['f1'] + results['ensemble_CICIDS']['f1'])/2:.4f}")
    print(f"Combined F1 (avg): {(results['combined_NSL-KDD']['f1'] + results['combined_CICIDS']['f1'])/2:.4f}")
    
    improvement = ((results['ensemble_NSL-KDD']['f1'] + results['ensemble_CICIDS']['f1'])/2 - 
                   (results['baseline_NSL-KDD']['f1'] + results['baseline_CICIDS']['f1'])/2)
    print(f"\nImprovement: {improvement:+.4f}")
    return results


if __name__ == '__main__':
    validate_phase1()
