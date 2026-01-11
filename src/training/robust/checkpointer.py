"""Enhanced checkpoint manager with RNG states and atomic writes."""
import os
import glob
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Set, List

import torch

from .utils import atomic_save, get_rng_state, set_rng_state
from .config import RobustTrainingConfig


class RobustCheckpointManager:
    """Enhanced checkpoint manager with full state preservation.
    
    Features:
    - Atomic writes (no corruption on crash)
    - RNG state preservation for reproducibility
    - Bad indices tracking
    - Sampler state for mid-epoch resume
    - Automatic cleanup of old checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        model_name: str,
        save_every_n_batches: int = 100,
        keep_n_checkpoints: int = 3,
        save_rng_state: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.save_every_n_batches = save_every_n_batches
        self.keep_n_checkpoints = keep_n_checkpoints
        self.save_rng_state = save_rng_state
        
        self.logger = logging.getLogger('RobustCheckpointManager')
    
    def _checkpoint_path(self, epoch: int, batch_idx: int) -> Path:
        return self.checkpoint_dir / f"{self.model_name}_e{epoch}_b{batch_idx}.pt"
    
    def save(
        self,
        epoch: int,
        batch_idx: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        scaler=None,
        global_step: int = 0,
        sampler_state: Optional[Dict] = None,
        bad_indices: Optional[Set[int]] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save checkpoint with full state."""
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        if self.save_rng_state:
            checkpoint['rng_state'] = get_rng_state()
        if sampler_state is not None:
            checkpoint['sampler_state'] = sampler_state
        if bad_indices:
            checkpoint['bad_indices'] = list(bad_indices)
        if metrics:
            checkpoint['metrics'] = metrics
        if extra:
            checkpoint.update(extra)
        
        path = self._checkpoint_path(epoch, batch_idx)
        atomic_save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: epoch {epoch}, batch {batch_idx}")
        
        self._cleanup_old_checkpoints()
        return path
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoints = self._list_checkpoints()
        if len(checkpoints) > self.keep_n_checkpoints:
            for old_path in checkpoints[:-self.keep_n_checkpoints]:
                try:
                    os.remove(old_path)
                    self.logger.debug(f"Removed old checkpoint: {old_path}")
                except OSError as e:
                    self.logger.warning(f"Failed to remove checkpoint {old_path}: {e}")
    
    def _list_checkpoints(self) -> List[Path]:
        """List all checkpoints sorted by modification time."""
        pattern = str(self.checkpoint_dir / f"{self.model_name}_e*_b*.pt")
        paths = [Path(p) for p in glob.glob(pattern)]
        return sorted(paths, key=lambda p: p.stat().st_mtime)
    
    def find_latest(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        checkpoints = self._list_checkpoints()
        return checkpoints[-1] if checkpoints else None
    
    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        scaler=None,
        device: str = 'cuda',
        restore_rng: bool = True,
    ) -> Dict[str, Any]:
        """Load latest checkpoint and restore state.
        
        Returns dict with: epoch, batch_idx, global_step, bad_indices, sampler_state, metrics
        """
        path = self.find_latest()
        if path is None:
            self.logger.info("No checkpoint found, starting fresh")
            return {
                'epoch': 0,
                'batch_idx': 0,
                'global_step': 0,
                'bad_indices': set(),
                'sampler_state': None,
                'metrics': {},
            }
        
        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if restore_rng and 'rng_state' in checkpoint:
            set_rng_state(checkpoint['rng_state'])
        
        result = {
            'epoch': checkpoint['epoch'],
            'batch_idx': checkpoint['batch_idx'],
            'global_step': checkpoint.get('global_step', 0),
            'bad_indices': set(checkpoint.get('bad_indices', [])),
            'sampler_state': checkpoint.get('sampler_state'),
            'metrics': checkpoint.get('metrics', {}),
        }
        
        self.logger.info(
            f"Resumed from epoch {result['epoch']}, batch {result['batch_idx']}"
        )
        return result
    
    def should_save(self, batch_idx: int) -> bool:
        """Check if checkpoint should be saved at this batch."""
        return batch_idx > 0 and batch_idx % self.save_every_n_batches == 0
    
    @classmethod
    def from_config(cls, config: RobustTrainingConfig, model_name: str) -> 'RobustCheckpointManager':
        """Create checkpoint manager from config."""
        return cls(
            checkpoint_dir=config.checkpoint_dir,
            model_name=model_name,
            save_every_n_batches=config.checkpoint_every_n_batches,
            keep_n_checkpoints=config.keep_n_checkpoints,
            save_rng_state=config.save_rng_state,
        )
