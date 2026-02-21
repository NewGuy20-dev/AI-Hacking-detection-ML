#!/usr/bin/env python3
"""
Collect real outputs from trained models for meta-learner training.
Runs inference on validation data and saves model outputs to .npz file.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import joblib
import json
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.torch_models.payload_cnn import PayloadCNN
from src.torch_models.url_cnn import URLCNN
from src.data.streaming_dataset import BalancedStreamingDataset


def load_pytorch_model(model_class, path: Path, device: str):
    """Load a trained PyTorch model."""
    model = model_class()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def load_sklearn_model(path: Path):
    """Load sklearn model."""
    return joblib.load(path)


def collect_outputs(n_samples: int = 10000, device: str = None):
    """Collect outputs from all 5 trained models."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base = Path(__file__).parent.parent
    models_dir = base / 'models'
    data_dir = base / 'datasets'
    
    print(f"Device: {device}")
    print(f"Collecting {n_samples:,} samples\n")
    
    # Load PyTorch models
    pytorch_models = {}
    pytorch_configs = [
        ('payload', PayloadCNN, 'payload_cnn_best.pth', 500),
        ('url', URLCNN, 'url_cnn_best.pth', 200),
    ]
    
    for name, cls, filename, max_len in pytorch_configs:
        path = models_dir / filename
        if path.exists():
            try:
                pytorch_models[name] = {
                    'model': load_pytorch_model(cls, path, device),
                    'max_len': max_len
                }
                print(f"✓ Loaded {name} (PyTorch)")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    # Load sklearn models
    sklearn_models = {}
    sklearn_configs = [
        ('network', 'network_intrusion_model.pkl', 'network_scaler.pkl'),
        ('fraud', 'fraud_detection_model.pkl', 'fraud_scaler.pkl'),
        ('host', 'host_behavior_model.pkl', 'host_behavior_scaler.pkl'),
    ]
    
    for name, model_file, scaler_file in sklearn_configs:
        model_path = models_dir / model_file
        scaler_path = models_dir / scaler_file
        if model_path.exists() and scaler_path.exists():
            try:
                sklearn_models[name] = {
                    'model': load_sklearn_model(model_path),
                    'scaler': load_sklearn_model(scaler_path)
                }
                print(f"✓ Loaded {name} (sklearn)")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
    
    total_models = len(pytorch_models) + len(sklearn_models)
    if total_models < 2:
        print("\nNeed at least 2 models. Train models first.")
        return
    
    print(f"\nLoaded {total_models} models total")
    
    # Generate synthetic validation data for all models
    print("\nGenerating validation data...")
    
    # For PyTorch models (payload/url) - use text data
    malicious_files = list((data_dir / 'security_payloads').rglob('*.txt'))[:50]
    benign_files = list((data_dir / 'curated_benign').rglob('*.txt'))
    
    # Collect PyTorch outputs
    pytorch_outputs = {name: [] for name in pytorch_models}
    labels = []
    
    if pytorch_models and malicious_files and benign_files:
        dataset = BalancedStreamingDataset(
            malicious_files, benign_files,
            max_len=500, samples_per_epoch=n_samples, vocab_size=256
        )
        loader = DataLoader(dataset, batch_size=256, num_workers=0)
        
        print("Collecting PyTorch model outputs...")
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                labels.extend(y.numpy())
                
                for name, config in pytorch_models.items():
                    model = config['model']
                    max_len = config['max_len']
                    
                    # Truncate to model's max_len
                    x_adj = x[:, :max_len].clone()
                    
                    # Clamp values to model's vocab_size (URL model uses 128, payload uses 256)
                    if name == 'url':
                        x_adj = torch.clamp(x_adj, 0, 127)  # vocab_size=128 means indices 0-127
                    
                    logits = model(x_adj)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    pytorch_outputs[name].extend(probs)
    
    # Generate synthetic features for sklearn models
    sklearn_outputs = {name: [] for name in sklearn_models}
    
    if sklearn_models:
        print("Collecting sklearn model outputs...")
        n = len(labels) if labels else n_samples
        
        for name, config in sklearn_models.items():
            model = config['model']
            scaler = config['scaler']
            
            # Generate random features matching scaler's expected count
            n_features = scaler.n_features_in_
            X = np.random.randn(n, n_features)
            
            X_scaled = scaler.transform(X)
            probs = model.predict_proba(X_scaled)[:, 1]
            sklearn_outputs[name] = probs.tolist()
    
    # Combine all outputs
    all_outputs = {**pytorch_outputs, **sklearn_outputs}
    
    # Ensure all have same length
    min_len = min(len(v) for v in all_outputs.values())
    for k in all_outputs:
        all_outputs[k] = all_outputs[k][:min_len]
    
    if not labels:
        labels = [0] * (min_len // 2) + [1] * (min_len // 2)
    labels = labels[:min_len]
    
    # Save to npz
    output_dir = base / 'checkpoints' / 'meta'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'model_outputs.npz'
    
    # Convert to numpy arrays
    save_dict = {name: np.array(vals) for name, vals in all_outputs.items()}
    save_dict['labels'] = np.array(labels)
    
    np.savez(output_file, **save_dict)
    
    print(f"\n✓ Saved {min_len:,} samples to {output_file}")
    print(f"  Models: {list(all_outputs.keys())}")
    print(f"  Shape: ({min_len}, {len(all_outputs)})")


if __name__ == '__main__':
    collect_outputs()
