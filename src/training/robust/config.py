"""Configuration for robust PyTorch training pipeline."""
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class RobustTrainingConfig:
    """Configuration for fault-tolerant training with mid-epoch checkpointing."""
    
    # Debug mode - forces num_workers=0 for easier debugging
    debug_mode: bool = False
    
    # Dataset safety settings
    sample_timeout_warning: float = 5.0  # Warn if sample takes longer (seconds)
    max_consecutive_failures: int = 10   # Raise error after N consecutive bad samples
    log_every_n_samples: int = 1000      # Log progress every N samples
    
    # DataLoader settings
    num_workers: int = 4
    dataloader_timeout: float = 120.0    # Timeout for batch loading (seconds)
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = False     # Safer default for recovery
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    checkpoint_every_n_batches: int = 100
    keep_n_checkpoints: int = 3
    save_rng_state: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    gpu_check_interval: float = 5.0      # Check GPU every N seconds
    gpu_idle_threshold: float = 5.0      # Warn if GPU util below this %
    gpu_idle_duration: float = 30.0      # Warn after idle for this long
    batch_time_warning_multiplier: float = 3.0  # Warn if batch > N * average
    
    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    
    # Recovery
    max_dataloader_recreations: int = 5  # Max times to recreate DataLoader per epoch
    skip_bad_batches: bool = True        # Continue training on batch errors
    
    def get_dataloader_kwargs(self) -> dict:
        """Get DataLoader kwargs based on config."""
        if self.debug_mode:
            return {
                'num_workers': 0,
                'pin_memory': False,
                'persistent_workers': False,
            }
        return {
            'num_workers': self.num_workers,
            'timeout': self.dataloader_timeout if self.num_workers > 0 else 0,
            'prefetch_factor': self.prefetch_factor if self.num_workers > 0 else None,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers and self.num_workers > 0,
        }
