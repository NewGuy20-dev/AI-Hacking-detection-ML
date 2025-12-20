"""PyTorch utilities for training and inference."""
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from pathlib import Path


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def setup_gpu(memory_limit_mb=3072):
    """Configure GPU for RTX 3050 (4GB VRAM)."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    # Print GPU info
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


class EarlyStopping:
    """Stop training when validation loss stops improving."""
    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    
    for batch in loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        with autocast('cuda'):
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
        
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / len(loader), correct / total


def save_model(model, path, example_input=None):
    """Save model as TorchScript for fast inference."""
    model.eval()
    path = Path(path)
    
    if example_input is not None:
        scripted = torch.jit.trace(model, example_input)
        scripted.save(str(path.with_suffix('.pt')))
    else:
        torch.save(model.state_dict(), str(path.with_suffix('.pth')))


def load_model(model_class, path, device, **kwargs):
    """Load model from checkpoint."""
    path = Path(path)
    
    if path.suffix == '.pt':
        return torch.jit.load(str(path)).to(device)
    else:
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(str(path), map_location=device))
        return model.to(device)
