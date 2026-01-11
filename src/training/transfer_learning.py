"""Transfer learning utilities for faster training."""
import torch
import torch.nn as nn
from torch.optim import AdamW
from typing import List, Optional


def freeze_embeddings(model: nn.Module):
    """Freeze embedding layer parameters."""
    for name, param in model.named_parameters():
        if 'embedding' in name.lower() or 'embed' in name.lower():
            param.requires_grad = False
    return count_trainable_params(model)


def unfreeze_embeddings(model: nn.Module):
    """Unfreeze embedding layer parameters."""
    for name, param in model.named_parameters():
        if 'embedding' in name.lower() or 'embed' in name.lower():
            param.requires_grad = True
    return count_trainable_params(model)


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """Freeze specific layers by name."""
    for name, param in model.named_parameters():
        if any(ln in name for ln in layer_names):
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    return count_trainable_params(model)


def count_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model: nn.Module) -> int:
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def get_layer_groups(model: nn.Module) -> dict:
    """Get parameter groups for differential learning rates."""
    groups = {'embedding': [], 'conv': [], 'fc': [], 'other': []}
    
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if 'embed' in name_lower:
            groups['embedding'].append(param)
        elif 'conv' in name_lower:
            groups['conv'].append(param)
        elif 'fc' in name_lower or 'linear' in name_lower:
            groups['fc'].append(param)
        else:
            groups['other'].append(param)
    
    return groups


def create_param_groups(model: nn.Module, base_lr: float, 
                       embed_lr_mult: float = 0.1) -> List[dict]:
    """Create parameter groups with differential learning rates."""
    groups = get_layer_groups(model)
    
    param_groups = []
    
    if groups['embedding']:
        param_groups.append({
            'params': groups['embedding'],
            'lr': base_lr * embed_lr_mult,
            'name': 'embedding'
        })
    
    if groups['conv']:
        param_groups.append({
            'params': groups['conv'],
            'lr': base_lr,
            'name': 'conv'
        })
    
    fc_params = groups['fc'] + groups['other']
    if fc_params:
        param_groups.append({
            'params': fc_params,
            'lr': base_lr,
            'name': 'fc'
        })
    
    return param_groups


def create_discriminative_optimizer(model: nn.Module, base_lr: float, 
                                    weight_decay: float = 1e-5,
                                    use_discriminative: bool = True) -> AdamW:
    """Create optimizer with discriminative learning rates."""
    if use_discriminative:
        param_groups = create_param_groups(model, base_lr, embed_lr_mult=0.1)
        return AdamW(param_groups, weight_decay=weight_decay)
    else:
        return AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)


class GradualUnfreezer:
    """Gradually unfreeze layers during training."""
    
    def __init__(self, model: nn.Module, unfreeze_schedule: dict):
        """
        Args:
            model: The model to manage
            unfreeze_schedule: Dict mapping epoch -> layers to unfreeze
                e.g., {0: [], 2: ['embedding'], 3: ['conv1']}
        """
        self.model = model
        self.schedule = unfreeze_schedule
        self.current_epoch = -1
    
    def step(self, epoch: int) -> bool:
        """Call at start of each epoch. Returns True if layers were unfrozen."""
        if epoch == self.current_epoch:
            return False
        
        self.current_epoch = epoch
        
        if epoch in self.schedule:
            layers = self.schedule[epoch]
            if layers == 'all':
                unfreeze_all(self.model)
                print(f"Epoch {epoch}: Unfroze all layers")
                return True
            elif layers:
                for name, param in self.model.named_parameters():
                    if any(ln in name for ln in layers):
                        param.requires_grad = True
                print(f"Epoch {epoch}: Unfroze layers {layers}")
                return True
        
        return False


def freeze_all_except(model: nn.Module, patterns: List[str]) -> int:
    """Freeze all parameters except those matching patterns. Returns trainable count."""
    count = 0
    for name, param in model.named_parameters():
        name_lower = name.lower()
        should_train = any(p in name_lower for p in patterns)
        param.requires_grad = should_train
        if should_train:
            count += param.numel()
    return count


class ProgressiveUnfreezer:
    """
    3-stage progressive unfreezing: FC → Conv → Embed
    
    Stage 1 (Epochs 0-1): Only FC layers trainable
    Stage 2 (Epoch 2): Add conv layers with 0.5x LR
    Stage 3 (Epoch 3+): Add embeddings with 0.1x LR
    """
    
    SCHEDULE = {
        0: (['fc', 'linear'], {'fc': 1.0}),
        2: (['fc', 'linear', 'conv'], {'fc': 1.0, 'conv': 0.5}),
        3: (['fc', 'linear', 'conv', 'embed', 'embedding'], {'fc': 1.0, 'conv': 0.5, 'embed': 0.1}),
    }
    
    def __init__(self, model: nn.Module, base_lr: float):
        self.model = model
        self.base_lr = base_lr
        self.current_epoch = -1
        self.current_lr_mults = {'fc': 1.0}
    
    def step(self, epoch: int):
        """
        Check if unfreezing needed at this epoch.
        Returns (changed: bool, lr_mults: dict or None)
        """
        if epoch == self.current_epoch or epoch not in self.SCHEDULE:
            return False, None
        
        self.current_epoch = epoch
        patterns, lr_mults = self.SCHEDULE[epoch]
        self.current_lr_mults = lr_mults
        
        count = freeze_all_except(self.model, patterns)
        print(f"\n🔓 Epoch {epoch}: Unfreezing {patterns}")
        print(f"   Trainable params: {count:,}")
        print(f"   LR multipliers: {lr_mults}")
        
        return True, lr_mults
    
    def create_optimizer(self, lr_mults: dict, weight_decay: float) -> AdamW:
        """Create optimizer with per-layer learning rates."""
        param_groups = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            name_lower = name.lower()
            
            # Determine LR multiplier based on layer type
            if 'embed' in name_lower:
                mult = lr_mults.get('embed', lr_mults.get('embedding', 0.1))
            elif 'conv' in name_lower:
                mult = lr_mults.get('conv', 0.5)
            else:
                mult = lr_mults.get('fc', 1.0)
            
            param_groups.append({
                'params': [param],
                'lr': self.base_lr * mult,
            })
        
        return AdamW(param_groups, weight_decay=weight_decay)


def get_transfer_learning_schedule(total_epochs: int, freeze_epochs: int = 2) -> dict:
    """
    Get default transfer learning schedule.
    
    Args:
        total_epochs: Total training epochs
        freeze_epochs: Number of epochs to keep embeddings frozen
    
    Returns:
        Schedule dict for GradualUnfreezer
    """
    schedule = {0: []}  # Start with embeddings frozen
    
    if freeze_epochs < total_epochs:
        schedule[freeze_epochs] = 'all'  # Unfreeze all after freeze_epochs
    
    return schedule


class WarmupScheduler:
    """Warmup wrapper for any PyTorch scheduler."""
    
    def __init__(self, optimizer, warmup_steps: int, base_scheduler):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_scheduler = base_scheduler
        self.current_step = 0
        
        # Store initial LRs
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
    
    def step(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr_scale = (self.current_step + 1) / self.warmup_steps
            for i, pg in enumerate(self.optimizer.param_groups):
                pg['lr'] = self.base_lrs[i] * lr_scale
        else:
            self.base_scheduler.step()
        self.current_step += 1
    
    def get_last_lr(self):
        if self.current_step < self.warmup_steps:
            lr_scale = (self.current_step + 1) / self.warmup_steps
            return [lr * lr_scale for lr in self.base_lrs]
        return self.base_scheduler.get_last_lr()
    
    def state_dict(self):
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
            'base_scheduler': self.base_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
