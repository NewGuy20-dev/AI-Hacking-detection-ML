"""Robust training pipeline for fault-tolerant PyTorch training.

This module provides components for:
- Fault-tolerant data loading with automatic recovery
- Mid-epoch checkpointing with RNG state preservation
- Resumable training from any checkpoint
- GPU utilization and batch timing monitoring

Usage:
    from training.robust import (
        RobustTrainingConfig,
        SafeDataset,
        ResumableSampler,
        RobustCheckpointManager,
        TrainingMonitor,
        RobustTrainer,
    )
    
    # Simple usage with RobustTrainer
    config = RobustTrainingConfig(checkpoint_every_n_batches=100)
    trainer = RobustTrainer(model, dataset, optimizer, criterion, config)
    trainer.fit(epochs=10, resume=True)
    
    # Or use components individually
    safe_dataset = SafeDataset(your_dataset)
    sampler = ResumableSampler(safe_dataset, shuffle=True)
    dataloader = DataLoader(safe_dataset, sampler=sampler, ...)
"""

from .config import RobustTrainingConfig
from .safe_dataset import SafeDataset, SkipBadIndicesDataset
from .resumable_sampler import ResumableSampler, ResumableDistributedSampler
from .checkpointer import RobustCheckpointManager
from .monitor import TrainingMonitor, GPUMonitor
from .trainer import RobustTrainer
from .utils import (
    setup_logger,
    atomic_save,
    get_rng_state,
    set_rng_state,
    BatchTimer,
    WorkerIndexTracker,
)

__all__ = [
    # Config
    'RobustTrainingConfig',
    # Dataset
    'SafeDataset',
    'SkipBadIndicesDataset',
    # Sampler
    'ResumableSampler',
    'ResumableDistributedSampler',
    # Checkpointing
    'RobustCheckpointManager',
    # Monitoring
    'TrainingMonitor',
    'GPUMonitor',
    # Trainer
    'RobustTrainer',
    # Utilities
    'setup_logger',
    'atomic_save',
    'get_rng_state',
    'set_rng_state',
    'BatchTimer',
    'WorkerIndexTracker',
]
