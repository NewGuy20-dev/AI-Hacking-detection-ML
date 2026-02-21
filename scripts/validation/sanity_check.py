"""Tier 1: Sanity Check - Verify models load and produce valid outputs."""
import sys
from pathlib import Path
import numpy as np
import torch
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.torch_models.payload_cnn import PayloadCNN
from src.torch_models.url_cnn import URLCNN
from src.torch_models.timeseries_lstm import TimeSeriesLSTM
from src.torch_models.meta_classifier import MetaClassifier


def check_pytorch_model(name, model_class, model_path, input_shape, device='cpu'):
    """Check a PyTorch model loads and predicts correctly."""
    try:
        model = model_class()
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        elif not isinstance(state, torch.jit.ScriptModule):
            model.load_state_dict(state)
        else:
            model = state
        model.to(device).eval()
        
        # Run 10 predictions
        with torch.no_grad():
            x = torch.randint(0, input_shape[1], (10, input_shape[0])).to(device)
            out = model(x)
            probs = torch.sigmoid(out).cpu().numpy()
        
        # Verify output
        if probs.shape[0] != 10:
            return False, f"Wrong output shape: {probs.shape}"
        if not (0 <= probs.min() <= probs.max() <= 1):
            return False, f"Output out of range [0,1]: [{probs.min()}, {probs.max()}]"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def check_sklearn_model(name, model_path, scaler_path, n_features):
    """Check a sklearn model loads and predicts correctly."""
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Run 10 predictions
        X = np.random.randn(10, n_features)
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        # Verify output
        if len(probs) != 10:
            return False, f"Wrong output shape: {len(probs)}"
        if not (0 <= probs.min() <= probs.max() <= 1):
            return False, f"Output out of range [0,1]: [{probs.min()}, {probs.max()}]"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_sanity_check(models_dir: Path, device: str = 'cpu') -> dict:
    """Run sanity check on all models."""
    results = {'passed': True, 'models': {}}
    
    print("\n" + "="*60)
    print("TIER 1: SANITY CHECK")
    print("="*60)
    
    # PyTorch models
    pytorch_models = [
        ('payload', PayloadCNN, 'payload_cnn_best.pth', (500, 256)),
        ('url', URLCNN, 'url_cnn_best.pth', (200, 128)),
        ('timeseries', TimeSeriesLSTM, 'timeseries_lstm.pt', (60, 8)),
        ('meta', MetaClassifier, 'meta_classifier.pt', (5, 1)),
    ]
    
    for name, cls, filename, shape in pytorch_models:
        path = models_dir / filename
        if path.exists():
            # Special handling for timeseries and meta
            if name == 'timeseries':
                try:
                    model = cls(input_dim=8, hidden_dim=128, num_layers=2)
                    state = torch.load(path, map_location=device, weights_only=False)
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        model.load_state_dict(state['model_state_dict'])
                    model.to(device).eval()
                    with torch.no_grad():
                        x = torch.randn(10, 60, 8).to(device)
                        out = model(x)
                        probs = torch.sigmoid(out).cpu().numpy()
                    passed = probs.shape[0] == 10 and 0 <= probs.min() <= probs.max() <= 1
                    msg = "OK" if passed else "Invalid output"
                except Exception as e:
                    passed, msg = False, str(e)
            elif name == 'meta':
                try:
                    model = cls(input_dim=5)
                    state = torch.load(path, map_location=device, weights_only=False)
                    if isinstance(state, dict) and 'model_state_dict' in state:
                        model.load_state_dict(state['model_state_dict'])
                    model.to(device).eval()
                    with torch.no_grad():
                        x = torch.rand(10, 5).to(device)
                        out = model(x)
                        probs = torch.sigmoid(out).cpu().numpy()
                    passed = probs.shape[0] == 10 and 0 <= probs.min() <= probs.max() <= 1
                    msg = "OK" if passed else "Invalid output"
                except Exception as e:
                    passed, msg = False, str(e)
            else:
                passed, msg = check_pytorch_model(name, cls, path, shape, device)
        else:
            passed, msg = False, "Model file not found"
        
        results['models'][name] = {'passed': passed, 'message': msg}
        status = "✓" if passed else "✗"
        print(f"  {name:15s}: {status} {msg}")
        if not passed:
            results['passed'] = False
    
    # Sklearn models
    sklearn_models = [
        ('network', 'network_intrusion_model.pkl', 'network_scaler.pkl'),
        ('fraud', 'fraud_detection_model.pkl', 'fraud_scaler.pkl'),
        ('host', 'host_behavior_model.pkl', 'host_behavior_scaler.pkl'),
    ]
    
    for name, model_file, scaler_file in sklearn_models:
        model_path = models_dir / model_file
        scaler_path = models_dir / scaler_file
        if model_path.exists() and scaler_path.exists():
            scaler = joblib.load(scaler_path)
            n_features = scaler.n_features_in_
            passed, msg = check_sklearn_model(name, model_path, scaler_path, n_features)
        else:
            passed, msg = False, "Model or scaler not found"
        
        results['models'][name] = {'passed': passed, 'message': msg}
        status = "✓" if passed else "✗"
        print(f"  {name:15s}: {status} {msg}")
        if not passed:
            results['passed'] = False
    
    print("-"*60)
    overall = "PASSED" if results['passed'] else "FAILED"
    print(f"Sanity Check: {overall}")
    
    return results
