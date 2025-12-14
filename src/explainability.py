"""Feature importance and model explainability."""
import joblib
import numpy as np
import json
from pathlib import Path


def get_feature_importance(model_path: str, feature_names: list) -> dict:
    """Extract feature importance from trained model."""
    model = joblib.load(model_path)
    
    # Handle different model types
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'model') and hasattr(model.model, 'estimators_'):
        # VotingClassifier - average across estimators
        importances = []
        for name, est in model.model.estimators_:
            if hasattr(est, 'feature_importances_'):
                importances.append(est.feature_importances_)
        importance = np.mean(importances, axis=0) if importances else None
    else:
        return {'error': 'Model does not support feature importance'}
    
    if importance is None:
        return {'error': 'Could not extract importance'}
    
    ranked = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return {
        'features': [{'name': n, 'importance': float(i)} for n, i in ranked],
        'top_5': [n for n, _ in ranked[:5]]
    }


def explain_prediction(model, X: np.ndarray, feature_names: list, idx: int = 0) -> dict:
    """Explain a single prediction."""
    proba = model.predict_proba(X[idx:idx+1])[0]
    pred = model.predict(X[idx:idx+1])[0]
    
    # Get feature values for this sample
    sample_features = {name: float(X[idx, i]) for i, name in enumerate(feature_names)}
    
    return {
        'prediction': int(pred),
        'confidence': float(max(proba)),
        'probabilities': {'normal': float(proba[0]), 'attack': float(proba[1])},
        'feature_values': sample_features
    }


def generate_importance_report():
    from train_unified import UNIFIED_FEATURES
    base = Path('/workspaces/AI-Hacking-detection-ML')
    
    report = {}
    
    # Network detector
    if (base / 'models/network_detector.pkl').exists():
        report['network_detector'] = get_feature_importance(
            str(base / 'models/network_detector.pkl'), UNIFIED_FEATURES
        )
    
    # Ensemble
    if (base / 'models/ensemble_voting.pkl').exists():
        report['ensemble_voting'] = get_feature_importance(
            str(base / 'models/ensemble_voting.pkl'), UNIFIED_FEATURES
        )
    
    # Save report
    with open(base / 'models/feature_importance.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Feature Importance Report:")
    for model_name, data in report.items():
        if 'top_5' in data:
            print(f"\n{model_name}: {data['top_5']}")
    
    return report


if __name__ == '__main__':
    generate_importance_report()
