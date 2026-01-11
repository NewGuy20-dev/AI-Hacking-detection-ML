#!/usr/bin/env python3
"""
Collect real outputs from trained models for meta-learner training.
Runs inference on validation data and saves model outputs to .npz file.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.torch_models.payload_cnn import PayloadCNN
from src.torch_models.url_cnn import URLCNN
from src.data.streaming_dataset import BalancedStreamingDataset


def load_model(model_class, path: Path, device: str):
    """Load a trained model from checkpoint."""
    model = model_class()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Handle both full checkpoint and state_dict only
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def collect_outputs(n_samples: int = 50000, device: str = None):
    """Collect outputs from all trained models on shared validation data."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    base = Path(__file__).parent.parent
    models_dir = base / 'models'
    
    print(f"Device: {device}")
    print(f"Collecting {n_samples:,} samples\n")
    
    # Load available models
    models = {}
    model_configs = [
        ('payload', PayloadCNN, 'payload_cnn_best.pt', 500, 256),
        ('url', URLCNN, 'url_cnn_best.pt', 200, 128),
    ]
    
    for name, cls, filename, max_len, vocab_size in model_configs:
        path = models_dir / filename
        if path.exists():
            try:
                models[name] = {
                    'model': load_model(cls, path, device),
                    'max_len': max_len,
                    'vocab_size': vocab_size
                }
                print(f"✓ Loaded {name} model from {filename}")
            except Exception as e:
                print(f"✗ Failed to load {name}: {e}")
        else:
            print(f"✗ {name} model not found: {filename}")
    
    if len(models) < 1:
        print("\nNo models loaded. Train models first.")
        return
    
    print(f"\nLoaded {len(models)} models: {list(models.keys())}")
    
    # Create validation dataset (use security payloads + curated benign)
    data_dir = base / 'datasets'
    
    malicious_files = list((data_dir / 'security_payloads').rglob('*.txt'))[:50]
    benign_files = list((data_dir / 'curated_benign').rglob('*.txt'))
    
    if not malicious_files or not benign_files:
        print("No data files found for validation")
        return
    
    print(f"Malicious files: {len(malicious_files)}")
    print(f"Benign files: {len(benign_files)}")
    
    # Collect outputs for each model
    all_outputs = {name: [] for name in models}
    all_labels = []
    
    # Use payload model's settings for shared dataset (longer max_len works for both)
    dataset = BalancedStreamingDataset(
        malicious_files, benign_files,
        max_len=500, samples_per_epoch=n_samples, vocab_size=256
    )
    
    loader = DataLoader(dataset, batch_size=256, num_workers=0)
    
    print(f"\nCollecting outputs...")
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            x = x.to(device)
            all_labels.extend(y.numpy())
            
            for name, config in models.items():
                model = config['model']
                
                # Truncate/adjust input for model's expected size
                max_len = config['max_len']
                vocab_size = config['vocab_size']
                
                x_adj = x[:, :max_len].clone()
                if vocab_size < 256:
                    x_adj = x_adj % vocab_size
                
                logits = model(x_adj)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_outputs[name].extend(probs)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * 256:,} samples")
    
    # Stack into matrix (N, num_models)
    model_names = sorted(models.keys())
    output_matrix = np.column_stack([all_outputs[name] for name in model_names])
    labels = np.array(all_labels, dtype=np.float32)
    
    print(f"\nOutput matrix shape: {output_matrix.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {labels.sum():.0f} malicious, {len(labels) - labels.sum():.0f} benign")
    
    # Save
    out_dir = base / 'checkpoints' / 'meta'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'model_outputs.npz'
    
    np.savez(
        out_path,
        outputs=output_matrix.astype(np.float32),
        labels=labels,
        model_names=model_names
    )
    
    print(f"\n✓ Saved to {out_path}")
    print(f"  Models: {model_names}")
    print(f"  Samples: {len(labels):,}")
    
    return output_matrix, labels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Collect model outputs for meta-learner')
    parser.add_argument('--samples', type=int, default=50000, help='Number of samples')
    args = parser.parse_args()
    
    collect_outputs(n_samples=args.samples)
