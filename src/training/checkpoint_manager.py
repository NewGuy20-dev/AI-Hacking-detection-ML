"""Checkpoint manager with auto-delete policy."""
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages checkpoints during training with auto-delete policy.
    
    Policy:
    - During training: Keep last N checkpoints
    - After training: Delete ALL checkpoints, keep only final models
    """
    
    def __init__(self, checkpoint_dir: Path, model_dir: Path, keep_last: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_dir = Path(model_dir)
        self.keep_last = keep_last
        self.checkpoints: Dict[str, list] = {}  # model_name -> list of checkpoint paths
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def clear_all_checkpoints(self):
        """Delete all checkpoints (call before new training run)."""
        if self.checkpoint_dir.exists():
            for f in self.checkpoint_dir.glob('*.pt'):
                f.unlink()
            print(f"Cleared all checkpoints from {self.checkpoint_dir}")
        self.checkpoints = {}
    
    def save_checkpoint(self, model: nn.Module, optimizer, scheduler, scaler,
                       epoch: int, batch: int, loss: float, model_name: str) -> Path:
        """Save checkpoint and rotate old ones."""
        filename = f"{model_name}_e{epoch}_b{batch}.pt"
        path = self.checkpoint_dir / filename
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
        }, path)
        
        # Track checkpoints per model
        if model_name not in self.checkpoints:
            self.checkpoints[model_name] = []
        self.checkpoints[model_name].append(path)
        
        # Rotate old checkpoints
        while len(self.checkpoints[model_name]) > self.keep_last:
            old = self.checkpoints[model_name].pop(0)
            if old.exists():
                old.unlink()
        
        return path
    
    def load_checkpoint(self, path: Path, model: nn.Module, 
                       optimizer=None, scheduler=None, scaler=None) -> Dict:
        """Load checkpoint into model and optionally optimizer/scheduler."""
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and checkpoint.get('scaler_state_dict'):
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint
    
    def get_latest_checkpoint(self, model_name: str) -> Optional[Path]:
        """Get most recent checkpoint for a model."""
        pattern = f"{model_name}_*.pt"
        checkpoints = sorted(self.checkpoint_dir.glob(pattern), 
                           key=lambda p: p.stat().st_mtime)
        return checkpoints[-1] if checkpoints else None
    
    def save_final_model(self, model: nn.Module, model_name: str):
        """Save final model weights (state dict and TorchScript)."""
        # State dict
        state_path = self.model_dir / f"{model_name}.pth"
        torch.save(model.state_dict(), state_path)
        
        # TorchScript for inference
        model.eval()
        try:
            scripted = torch.jit.script(model)
            script_path = self.model_dir / f"{model_name}.pt"
            scripted.save(str(script_path))
            print(f"Saved TorchScript model: {script_path}")
        except Exception as e:
            print(f"Could not save TorchScript (will use state dict): {e}")
        
        print(f"Saved final model: {state_path}")
        return state_path
    
    def cleanup_after_training(self, model_name: str):
        """Delete all checkpoints for a model after training completes."""
        pattern = f"{model_name}_*.pt"
        deleted = 0
        for f in self.checkpoint_dir.glob(pattern):
            f.unlink()
            deleted += 1
        
        if model_name in self.checkpoints:
            del self.checkpoints[model_name]
        
        if deleted > 0:
            print(f"Deleted {deleted} checkpoints for {model_name}")
    
    def cleanup_all_after_training(self):
        """Delete ALL checkpoints after full training run completes."""
        deleted = 0
        for f in self.checkpoint_dir.glob('*.pt'):
            f.unlink()
            deleted += 1
        
        self.checkpoints = {}
        
        if deleted > 0:
            print(f"Deleted all {deleted} checkpoints")
