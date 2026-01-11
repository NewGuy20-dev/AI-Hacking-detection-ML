"""Train Meta-Classifier that combines all model outputs."""
import sys
import argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.amp import GradScaler
from tqdm import tqdm

from torch_models.meta_classifier import MetaClassifier
from torch_models.utils import setup_gpu, EarlyStopping, save_model
from training.checkpoint import CheckpointManager


def generate_model_outputs(n_samples=20000):
    """
    Simulate outputs from 5 models:
    - network_prob: Network intrusion model (sklearn)
    - fraud_prob: Fraud detection model (sklearn)  
    - url_prob: URL CNN + LightGBM hybrid
    - payload_prob: Payload CNN
    - timeseries_prob: Time-series LSTM
    """
    # Ground truth labels
    labels = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    outputs = []
    for label in labels:
        if label == 1:  # Attack
            # Models should generally predict high, but with variation
            network = np.clip(np.random.beta(5, 2), 0, 1)      # Skewed high
            fraud = np.clip(np.random.beta(3, 2), 0, 1)        # Moderate high
            url = np.clip(np.random.beta(4, 2), 0, 1)          # High
            payload = np.clip(np.random.beta(4, 3), 0, 1)      # Moderate
            timeseries = np.clip(np.random.beta(5, 2), 0, 1)   # High
            
            # Sometimes models disagree (realistic scenario)
            if np.random.random() < 0.2:
                # One model misses
                idx = np.random.randint(5)
                vals = [network, fraud, url, payload, timeseries]
                vals[idx] = np.random.beta(2, 5)  # Low score
                network, fraud, url, payload, timeseries = vals
                
        else:  # Normal
            # Models should generally predict low
            network = np.clip(np.random.beta(2, 5), 0, 1)
            fraud = np.clip(np.random.beta(2, 4), 0, 1)
            url = np.clip(np.random.beta(2, 5), 0, 1)
            payload = np.clip(np.random.beta(2, 4), 0, 1)
            timeseries = np.clip(np.random.beta(2, 5), 0, 1)
            
            # Sometimes false positives
            if np.random.random() < 0.15:
                idx = np.random.randint(5)
                vals = [network, fraud, url, payload, timeseries]
                vals[idx] = np.random.beta(5, 2)  # High score (false positive)
                network, fraud, url, payload, timeseries = vals
        
        outputs.append([network, fraud, url, payload, timeseries])
    
    return np.array(outputs, dtype=np.float32), labels


def load_hybrid_data(base_path: Path, n_synthetic: int = 10000):
    """
    Load hybrid training data: 70% real model outputs + 30% synthetic.
    
    Args:
        base_path: Project root path
        n_synthetic: Number of synthetic samples to generate
    
    Returns:
        outputs: (N, num_models) array of model probabilities
        labels: (N,) array of ground truth labels
    """
    real_path = base_path / 'checkpoints' / 'meta' / 'model_outputs.npz'
    
    # Generate synthetic data
    syn_outputs, syn_labels = generate_model_outputs(n_synthetic)
    
    if real_path.exists():
        print(f"Loading real model outputs from {real_path}")
        data = np.load(real_path)
        real_outputs = data['outputs']
        real_labels = data['labels']
        
        # Pad real outputs to 5 columns if needed (we may only have 2 models)
        if real_outputs.shape[1] < 5:
            n_pad = 5 - real_outputs.shape[1]
            # Fill missing model outputs with 0.5 (neutral)
            padding = np.full((real_outputs.shape[0], n_pad), 0.5, dtype=np.float32)
            real_outputs = np.hstack([real_outputs, padding])
        
        # Combine: use all real + 30% synthetic
        n_real = len(real_labels)
        n_use_syn = max(n_real // 2, 5000)  # At least 5k synthetic
        
        # Random sample from synthetic
        idx = np.random.choice(len(syn_labels), min(n_use_syn, len(syn_labels)), replace=False)
        
        outputs = np.vstack([real_outputs, syn_outputs[idx]])
        labels = np.concatenate([real_labels, syn_labels[idx]])
        
        print(f"Hybrid data: {n_real} real + {len(idx)} synthetic = {len(labels)} total")
        print(f"Real/Synthetic ratio: {n_real / len(labels) * 100:.1f}% / {len(idx) / len(labels) * 100:.1f}%")
    else:
        print(f"No real outputs found at {real_path}")
        print("Using synthetic data only. Run collect_model_outputs.py first for hybrid training.")
        outputs, labels = syn_outputs, syn_labels
    
    return outputs.astype(np.float32), labels.astype(np.float32)


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint-every', type=int, default=500, help='Save checkpoint every N batches')
    parser.add_argument('--hybrid', action='store_true', default=True, help='Use hybrid real+synthetic data')
    args = parser.parse_args()
    
    device = setup_gpu()
    base_path = Path(__file__).parent.parent.parent
    
    # Checkpoint manager
    ckpt_dir = base_path / 'checkpoints' / 'meta'
    ckpt_mgr = CheckpointManager(str(ckpt_dir), 'meta_classifier', args.checkpoint_every)
    
    # Load hybrid data (real + synthetic) or synthetic only
    print("\n--- Loading Training Data ---")
    if args.hybrid:
        model_outputs, labels = load_hybrid_data(base_path, n_synthetic=15000)
    else:
        model_outputs, labels = generate_model_outputs(30000)
    
    print(f"Total: {len(labels)} samples ({sum(labels==0):.0f} normal, {sum(labels==1):.0f} attack)")
    print(f"Input shape: {model_outputs.shape} (samples, 5 model scores)")
    
    # Create dataset
    X = torch.tensor(model_outputs, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    # Use num_workers=0 on Windows to avoid issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    print("\n--- Creating Model ---")
    model = MetaClassifier(num_models=5).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler = GradScaler()
    early_stop = EarlyStopping(patience=7)
    
    # Resume from checkpoint if requested
    start_epoch, start_batch, global_step = 0, 0, 0
    if args.resume or ckpt_mgr.find_latest():
        resume_info = ckpt_mgr.load(model, optimizer, scheduler, scaler, device)
        start_epoch = resume_info['epoch']
        start_batch = resume_info['batch_idx']
        global_step = resume_info['global_step']
        if start_batch >= len(train_loader):
            start_epoch += 1
            start_batch = 0
    
    # Training loop
    print("\n--- Training ---")
    best_val_acc = 0
    best_state = None
    
    for epoch in range(start_epoch, 50):
        # Train
        model.train()
        train_loss = 0
        batches_processed = 0
        
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    initial=epoch_start_batch, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (inputs, targets) in pbar:
            if batch_idx < epoch_start_batch:
                continue
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            batches_processed += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if ckpt_mgr.should_save(batch_idx):
                ckpt_mgr.save(epoch, batch_idx, model, optimizer, scheduler, scaler, global_step)
        
        ckpt_mgr.save(epoch, len(train_loader), model, optimizer, scheduler, scaler, global_step)
        
        train_loss /= max(batches_processed, 1)
        
        # Validate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        if early_stop(val_loss):
            print("Early stopping triggered")
            break
    
    # Save best model
    print("\n--- Saving Model ---")
    model.load_state_dict(best_state)
    models_dir = base_path / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model.eval()
    example = torch.zeros(1, 5, dtype=torch.float32).to(device)
    save_model(model, models_dir / 'meta_classifier', example)
    torch.save(best_state, models_dir / 'meta_classifier.pth')
    
    print(f"✓ Model saved to models/meta_classifier.pt")
    print(f"✓ Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    train()
