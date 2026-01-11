"""Mixed precision trainer with gradient scaling for fast training."""
import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader


class AMPTrainer:
    """Automatic Mixed Precision trainer for 2-3x speedup."""
    
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module = None,
                 device: str = None,
                 accumulation_steps: int = 1,
                 max_grad_norm: float = 1.0):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.scaler = GradScaler() if self.device == 'cuda' else None
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader, epoch: int = 0) -> Dict[str, float]:
        """Train for one epoch with mixed precision."""
        self.model.train()
        self.epoch = epoch
        
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Mixed precision forward pass
            if self.device == 'cuda':
                with autocast():
                    outputs = self.model(inputs).squeeze()
                    loss = self.criterion(outputs, targets) / self.accumulation_steps
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets) / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            self.global_step += 1
        
        elapsed = time.time() - start_time
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total if total > 0 else 0,
            'samples_per_sec': total / elapsed,
            'elapsed': elapsed,
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            if self.device == 'cuda':
                with autocast():
                    outputs = self.model(inputs).squeeze()
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs).squeeze()
                loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            all_preds.extend(probs.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
        
        # Calculate FP/FN rates
        fp = sum(1 for p, t in zip(all_preds, all_targets) if p > 0.5 and t == 0)
        fn = sum(1 for p, t in zip(all_preds, all_targets) if p <= 0.5 and t == 1)
        total_neg = sum(1 for t in all_targets if t == 0)
        total_pos = sum(1 for t in all_targets if t == 1)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total if total > 0 else 0,
            'fp_rate': fp / total_neg if total_neg > 0 else 0,
            'fn_rate': fn / total_pos if total_pos > 0 else 0,
        }
    
    def save_checkpoint(self, path: Path, metrics: Dict[str, Any] = None):
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics or {},
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint.get('metrics', {})


def create_optimizer(model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-5):
    """Create AdamW optimizer with weight decay."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = 0.1):
    """Create OneCycleLR scheduler for faster convergence."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.defaults['lr'] * 10,
        total_steps=num_training_steps,
        pct_start=warmup_ratio,
        anneal_strategy='cos',
    )
