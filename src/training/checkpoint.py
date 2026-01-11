"""Mid-epoch checkpoint utility for robust PyTorch training on Windows."""
import os
import glob
import torch
from pathlib import Path
from typing import Optional, Dict, Any

class CheckpointManager:
    """Manages mid-epoch checkpoints with save/load/resume logic."""
    
    def __init__(self, checkpoint_dir: str, model_name: str, save_every_n_batches: int = 500):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.save_every_n_batches = save_every_n_batches
    
    def _checkpoint_path(self, epoch: int, batch_idx: int) -> Path:
        return self.checkpoint_dir / f"{self.model_name}_epoch{epoch}_batch{batch_idx}.pt"
    
    def save(self, epoch: int, batch_idx: int, model: torch.nn.Module,
             optimizer: torch.optim.Optimizer, scheduler=None, scaler=None,
             global_step: int = 0, extra: Dict[str, Any] = None):
        """Save mid-epoch checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        if extra:
            checkpoint.update(extra)
        
        path = self._checkpoint_path(epoch, batch_idx)
        torch.save(checkpoint, path)
        print(f"Saved checkpoint at epoch {epoch} batch {batch_idx}")
        
        # Keep only last 3 checkpoints to save disk space
        self._cleanup_old_checkpoints(keep=3)
    
    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Remove old checkpoints, keeping only the most recent ones."""
        pattern = str(self.checkpoint_dir / f"{self.model_name}_epoch*_batch*.pt")
        checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
        for old_ckpt in checkpoints[:-keep]:
            os.remove(old_ckpt)
    
    def find_latest(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        pattern = str(self.checkpoint_dir / f"{self.model_name}_epoch*_batch*.pt")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None
        return Path(max(checkpoints, key=os.path.getmtime))
    
    def load(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
             scheduler=None, scaler=None, device='cuda') -> Dict[str, Any]:
        """Load latest checkpoint and return resume info."""
        path = self.find_latest()
        if path is None:
            return {'epoch': 0, 'batch_idx': 0, 'global_step': 0}
        
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint['epoch']
        batch_idx = checkpoint['batch_idx']
        global_step = checkpoint.get('global_step', 0)
        
        print(f"Resuming training from epoch {epoch}, batch {batch_idx}")
        return {'epoch': epoch, 'batch_idx': batch_idx, 'global_step': global_step}
    
    def should_save(self, batch_idx: int) -> bool:
        """Check if checkpoint should be saved at this batch."""
        return batch_idx > 0 and batch_idx % self.save_every_n_batches == 0


def train_epoch_with_checkpoints(
    model, train_loader, optimizer, criterion, scaler, scheduler,
    device, epoch: int, ckpt_mgr: CheckpointManager,
    start_batch_idx: int = 0, global_step: int = 0
) -> tuple:
    """Training loop with mid-epoch checkpointing and resume support.
    
    Returns: (avg_loss, final_global_step)
    """
    model.train()
    total_loss = 0.0
    batches_processed = 0
    
    from tqdm import tqdm
    total_batches = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=total_batches, 
                initial=start_batch_idx, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in pbar:
        # Skip already-processed batches on resume
        if batch_idx < start_batch_idx:
            continue
        
        # Unpack batch (assumes (inputs, labels) format)
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        batches_processed += 1
        global_step += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Mid-epoch checkpoint
        if ckpt_mgr.should_save(batch_idx):
            ckpt_mgr.save(epoch, batch_idx, model, optimizer, scheduler, scaler, global_step)
    
    # End-of-epoch checkpoint
    ckpt_mgr.save(epoch, total_batches, model, optimizer, scheduler, scaler, global_step)
    
    avg_loss = total_loss / max(batches_processed, 1)
    return avg_loss, global_step
