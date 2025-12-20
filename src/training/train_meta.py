"""Train Meta-Classifier that combines all model outputs."""
import sys
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


def train():
    """Main training function."""
    device = setup_gpu()
    base_path = Path(__file__).parent.parent.parent
    
    # Generate simulated model outputs
    print("\n--- Generating Model Outputs ---")
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
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    
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
    
    # Training loop
    print("\n--- Training ---")
    best_val_acc = 0
    best_state = None
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
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
        
        train_loss /= len(train_loader)
        
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
