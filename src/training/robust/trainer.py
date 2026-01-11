"""Robust trainer integrating all fault-tolerant components."""
import signal
import time
import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import RobustTrainingConfig
from .safe_dataset import SafeDataset
from .resumable_sampler import ResumableSampler
from .checkpointer import RobustCheckpointManager
from .monitor import TrainingMonitor
from .utils import setup_logger


class RobustTrainer:
    """Fault-tolerant trainer with automatic recovery and mid-epoch checkpointing.
    
    Features:
    - Wraps dataset with SafeDataset for fault tolerance
    - Uses ResumableSampler for mid-epoch resume
    - Automatic checkpoint saving with RNG state
    - DataLoader timeout recovery with recreation
    - Background GPU/batch monitoring
    - SIGINT handler for graceful shutdown
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Optional[RobustTrainingConfig] = None,
        scheduler=None,
        scaler=None,
        val_dataset: Optional[Dataset] = None,
        model_name: str = "model",
        device: str = "cuda",
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config or RobustTrainingConfig()
        self.model_name = model_name
        self.device = device
        self.collate_fn = collate_fn
        
        # Setup logging
        self.logger = setup_logger(
            'RobustTrainer',
            self.config.log_dir if self.config.log_to_file else None,
            self.config.log_level,
            self.config.log_to_file
        )
        
        # Initialize components
        self.checkpoint_mgr = RobustCheckpointManager.from_config(config, model_name)
        self.monitor = TrainingMonitor(config)
        
        # Wrap dataset with SafeDataset
        self.safe_dataset = SafeDataset(
            train_dataset,
            timeout_warning=config.sample_timeout_warning,
            max_consecutive_failures=config.max_consecutive_failures,
            log_every_n=config.log_every_n_samples,
        )
        
        # State
        self._dataloader: Optional[DataLoader] = None
        self._sampler: Optional[ResumableSampler] = None
        self._interrupted = False
        self._dataloader_recreations = 0
        self.global_step = 0
        self.bad_indices = set()
        
        # Setup signal handler
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_interrupt)
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        if self._interrupted:
            # Second interrupt - force exit
            self.logger.warning("Force exit requested")
            signal.signal(signal.SIGINT, self._original_sigint)
            raise KeyboardInterrupt
        
        self._interrupted = True
        self.logger.info("Interrupt received, saving checkpoint and exiting...")
    
    def _create_dataloader(self, start_index: int = 0, epoch: int = 0) -> DataLoader:
        """Create DataLoader with ResumableSampler."""
        self._sampler = ResumableSampler(
            self.safe_dataset,
            shuffle=True,
            seed=42 + epoch,
            start_index=start_index,
        )
        
        kwargs = self.config.get_dataloader_kwargs()
        
        return DataLoader(
            self.safe_dataset,
            batch_size=64,  # Can be overridden
            sampler=self._sampler,
            collate_fn=self.collate_fn,
            **kwargs
        )
    
    def _recreate_dataloader(self, epoch: int, batch_idx: int) -> bool:
        """Recreate DataLoader after timeout/error."""
        self._dataloader_recreations += 1
        
        if self._dataloader_recreations > self.config.max_dataloader_recreations:
            self.logger.error(
                f"Too many DataLoader recreations ({self._dataloader_recreations})"
            )
            return False
        
        self.logger.warning(
            f"Recreating DataLoader (attempt {self._dataloader_recreations})"
        )
        
        # Collect bad indices from SafeDataset
        self.bad_indices.update(self.safe_dataset.get_bad_indices())
        
        # Delete old dataloader
        del self._dataloader
        self._dataloader = None
        
        # Create new one starting from next batch
        self._dataloader = self._create_dataloader(batch_idx + 1, epoch)
        return True
    
    def train_epoch(
        self,
        epoch: int,
        start_batch: int = 0,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """Train for one epoch with fault tolerance."""
        self.model.train()
        self.monitor.set_training_active(True)
        self._dataloader_recreations = 0
        
        # Create dataloader
        self._dataloader = self._create_dataloader(start_batch, epoch)
        total_batches = self._sampler.total_length // batch_size
        
        total_loss = 0.0
        batches_processed = 0
        
        pbar = tqdm(
            total=total_batches,
            initial=start_batch,
            desc=f"Epoch {epoch + 1}"
        )
        
        batch_idx = start_batch
        iterator = iter(self._dataloader)
        
        while batch_idx < total_batches:
            if self._interrupted:
                break
            
            batch_start = time.time()
            
            try:
                batch = next(iterator)
            except StopIteration:
                break
            except RuntimeError as e:
                if "timeout" in str(e).lower():
                    self.logger.error(f"DataLoader timeout at batch {batch_idx}")
                    if not self._recreate_dataloader(epoch, batch_idx):
                        break
                    iterator = iter(self._dataloader)
                    continue
                raise
            
            # Save checkpoint BEFORE processing (so we can skip on resume)
            if self.checkpoint_mgr.should_save(batch_idx):
                self._save_checkpoint(epoch, batch_idx)
            
            try:
                loss = self._train_step(batch)
                total_loss += loss
                batches_processed += 1
                self.global_step += 1
                
                # Record batch time
                batch_time = time.time() - batch_start
                self.monitor.record_batch_time(batch_idx, batch_time)
                
                pbar.set_postfix({'loss': f'{loss:.4f}'})
                pbar.update(1)
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                if not self.config.skip_bad_batches:
                    raise
            
            batch_idx += 1
        
        pbar.close()
        self.monitor.set_training_active(False)
        
        # End of epoch checkpoint
        self._save_checkpoint(epoch, batch_idx)
        
        # Collect bad indices
        self.bad_indices.update(self.safe_dataset.get_bad_indices())
        
        return {
            'loss': total_loss / max(batches_processed, 1),
            'batches': batches_processed,
            'bad_samples': len(self.bad_indices),
        }
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single training step."""
        inputs = batch['input'].to(self.device)
        targets = batch['target'].to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler:
            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), targets)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs.squeeze(), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return loss.item()
    
    def _save_checkpoint(self, epoch: int, batch_idx: int) -> None:
        """Save checkpoint with full state."""
        sampler_state = self._sampler.state_dict() if self._sampler else None
        
        self.checkpoint_mgr.save(
            epoch=epoch,
            batch_idx=batch_idx,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            global_step=self.global_step,
            sampler_state=sampler_state,
            bad_indices=self.bad_indices,
        )
    
    def resume(self) -> Dict[str, Any]:
        """Resume from latest checkpoint."""
        result = self.checkpoint_mgr.load(
            self.model,
            self.optimizer,
            self.scheduler,
            self.scaler,
            self.device,
        )
        
        self.global_step = result['global_step']
        self.bad_indices = result['bad_indices']
        
        return result
    
    def fit(
        self,
        epochs: int,
        batch_size: int = 64,
        resume: bool = False,
        val_fn: Optional[Callable[[], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Full training loop with optional resume."""
        start_epoch, start_batch = 0, 0
        
        if resume:
            result = self.resume()
            start_epoch = result['epoch']
            start_batch = result['batch_idx']
            
            # If epoch was complete, move to next
            total_batches = len(self.train_dataset) // batch_size
            if start_batch >= total_batches:
                start_epoch += 1
                start_batch = 0
        
        self.monitor.start()
        history = {'train_loss': [], 'val_metrics': []}
        
        try:
            for epoch in range(start_epoch, epochs):
                if self._interrupted:
                    break
                
                # Determine start batch for this epoch
                epoch_start = start_batch if epoch == start_epoch else 0
                
                # Train
                metrics = self.train_epoch(epoch, epoch_start, batch_size)
                history['train_loss'].append(metrics['loss'])
                
                self.logger.info(
                    f"Epoch {epoch + 1}: loss={metrics['loss']:.4f}, "
                    f"batches={metrics['batches']}, bad_samples={metrics['bad_samples']}"
                )
                
                # Validate
                if val_fn:
                    val_metrics = val_fn()
                    history['val_metrics'].append(val_metrics)
                    self.logger.info(f"Validation: {val_metrics}")
                
                # Step scheduler
                if self.scheduler:
                    if hasattr(self.scheduler, 'step'):
                        self.scheduler.step(metrics['loss'])
        
        finally:
            self.monitor.stop()
            signal.signal(signal.SIGINT, self._original_sigint)
        
        return {
            'history': history,
            'bad_indices': self.bad_indices,
            'global_step': self.global_step,
            'interrupted': self._interrupted,
        }
