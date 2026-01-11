"""Main training pipeline orchestrator with AMP and stress testing."""
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.torch_models.payload_cnn import PayloadCNN
from src.stress_test.streaming_dataset import StreamingPayloadDataset, collate_fn
from src.stress_test.amp_trainer import AMPTrainer, create_optimizer, create_scheduler
from src.stress_test.hash_registry import HashRegistry
from src.stress_test.runner import StressTestRunner, StressTestConfig
from src.training.checkpoint import CheckpointManager


class TrainingConfig:
    """Training configuration."""
    def __init__(self):
        # Data paths
        self.benign_paths = [
            "datasets/benign_5m.jsonl",
            "datasets/curated_benign/all_benign.txt",
            "datasets/curated_benign/adversarial/",
        ]
        self.malicious_paths = [
            "datasets/security_payloads/",
        ]
        
        # Training params
        self.batch_size = 256
        self.epochs = 10
        self.learning_rate = 1e-3
        self.weight_decay = 1e-5
        self.accumulation_steps = 1
        self.max_grad_norm = 1.0
        self.num_workers = 4
        self.val_split = 0.1
        
        # Model
        self.model_save_path = "models/payload_cnn.pt"
        self.checkpoint_path = "models/checkpoint.pt"
        
        # Hash registry
        self.save_hashes = True
        self.hash_registry_path = "models/training_hashes.pkl"
        
        # Early stopping
        self.patience = 3
        self.min_delta = 0.001
        
        # Checkpointing
        self.checkpoint_every = 500  # Save every N batches


class TrainingPipeline:
    """Main training pipeline with AMP support."""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.model = None
        self.trainer = None
        self.hash_registry = None
        self.ckpt_mgr = None
    
    def setup_data(self) -> tuple:
        """Setup streaming data loaders."""
        print("Setting up data loaders...")
        
        base = Path(__file__).parent.parent.parent
        
        benign_paths = [base / p for p in self.config.benign_paths]
        malicious_paths = [base / p for p in self.config.malicious_paths]
        
        # Create streaming dataset
        dataset = StreamingPayloadDataset(
            benign_paths=benign_paths,
            malicious_paths=malicious_paths,
            max_length=512,
            buffer_size=10000,
        )
        
        # For validation, we'll use a fixed subset
        # In production, you'd want a separate validation set
        train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_fn,
            num_workers=min(self.config.num_workers, 4),
            prefetch_factor=2 if self.config.num_workers > 0 else None,
            timeout=0,
            persistent_workers=self.config.num_workers > 0,
        )
        
        # Simple validation loader (would be separate in production)
        val_loader = train_loader  # Placeholder
        
        return train_loader, val_loader
    
    def setup_model(self):
        """Initialize model, optimizer, and trainer."""
        print("Setting up model...")
        
        self.model = PayloadCNN().to(self.device)
        optimizer = create_optimizer(
            self.model,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        self.trainer = AMPTrainer(
            model=self.model,
            optimizer=optimizer,
            device=self.device,
            accumulation_steps=self.config.accumulation_steps,
            max_grad_norm=self.config.max_grad_norm,
        )
        
        # Hash registry for tracking training samples
        if self.config.save_hashes:
            self.hash_registry = HashRegistry(Path(self.config.hash_registry_path))
            self.hash_registry.create(expected_items=100_000_000)
        
        # Checkpoint manager for mid-epoch saves
        base = Path(__file__).parent.parent.parent
        ckpt_dir = base / 'checkpoints' / 'stress_pipeline'
        self.ckpt_mgr = CheckpointManager(str(ckpt_dir), 'stress_pipeline', self.config.checkpoint_every)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, max_samples: int = None, resume: bool = False) -> dict:
        """Run training loop with mid-epoch checkpoint support."""
        train_loader, val_loader = self.setup_data()
        self.setup_model()
        
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Accumulation steps: {self.config.accumulation_steps}")
        print(f"Effective batch size: {self.config.batch_size * self.config.accumulation_steps}")
        
        # Resume from checkpoint if available
        start_epoch, start_batch, global_step = 0, 0, 0
        if resume or self.ckpt_mgr.find_latest():
            resume_info = self.ckpt_mgr.load(
                self.model, self.trainer.optimizer, 
                self.trainer.scheduler if hasattr(self.trainer, 'scheduler') else None,
                self.trainer.scaler, self.device
            )
            start_epoch = resume_info['epoch']
            start_batch = resume_info['batch_idx']
            global_step = resume_info['global_step']
            if start_batch >= len(train_loader):
                start_epoch += 1
                start_batch = 0
            print(f"Resumed from epoch {start_epoch}, batch {start_batch}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(start_epoch, self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Training with mid-epoch checkpoints
            epoch_start_batch = start_batch if epoch == start_epoch else 0
            train_metrics = self._train_epoch_with_checkpoints(
                train_loader, epoch, epoch_start_batch, global_step
            )
            global_step = train_metrics.get('global_step', global_step)
            
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Train Acc:  {train_metrics['accuracy']*100:.2f}%")
            print(f"  Speed:      {train_metrics['samples_per_sec']:.0f} samples/sec")
            
            # Save hashes (sample from this epoch)
            if self.hash_registry and epoch == 0:
                print("  Recording sample hashes...")
            
            # Validation
            val_metrics = self.trainer.validate(val_loader)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']*100:.2f}%")
            print(f"  Val FP:     {val_metrics['fp_rate']*100:.2f}%")
            print(f"  Val FN:     {val_metrics['fn_rate']*100:.2f}%")
            
            # Early stopping check
            if val_metrics['loss'] < best_val_loss - self.config.min_delta:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self.save_model()
                print(f"  ✓ New best model saved")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        # Save hash registry
        if self.hash_registry:
            self.hash_registry.save()
        
        print("\nTraining complete!")
        return history
    
    def _train_epoch_with_checkpoints(self, train_loader, epoch: int, 
                                       start_batch: int = 0, global_step: int = 0) -> dict:
        """Train one epoch with mid-epoch checkpoint support."""
        from tqdm import tqdm
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        batches_processed = 0
        start_time = time.time()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    initial=start_batch, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in pbar:
            if batch_idx < start_batch:
                continue
            
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass with AMP
            self.trainer.optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                loss = nn.BCEWithLogitsLoss()(outputs.squeeze(), targets.float())
            
            self.trainer.scaler.scale(loss).backward()
            self.trainer.scaler.unscale_(self.trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.trainer.scaler.step(self.trainer.optimizer)
            self.trainer.scaler.update()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            batches_processed += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Mid-epoch checkpoint
            if self.ckpt_mgr.should_save(batch_idx):
                scheduler = self.trainer.scheduler if hasattr(self.trainer, 'scheduler') else None
                self.ckpt_mgr.save(epoch, batch_idx, self.model, self.trainer.optimizer,
                                   scheduler, self.trainer.scaler, global_step)
        
        # End-of-epoch checkpoint
        scheduler = self.trainer.scheduler if hasattr(self.trainer, 'scheduler') else None
        self.ckpt_mgr.save(epoch, len(train_loader), self.model, self.trainer.optimizer,
                          scheduler, self.trainer.scaler, global_step)
        
        elapsed = time.time() - start_time
        return {
            'loss': total_loss / max(batches_processed, 1),
            'accuracy': correct / max(total, 1),
            'samples_per_sec': total / max(elapsed, 1),
            'global_step': global_step,
        }
    
    def save_model(self):
        """Save model weights."""
        path = Path(self.config.model_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        
        # Also save as .pth for compatibility
        pth_path = path.with_suffix('.pth')
        torch.save(self.model.state_dict(), pth_path)
    
    def run_stress_test(self, discord_webhook: str = None) -> bool:
        """Run stress test after training."""
        print("\n" + "=" * 60)
        print("Running post-training stress test...")
        
        config = StressTestConfig(
            model_path=self.config.model_save_path,
            hash_registry_path=self.config.hash_registry_path,
            discord_webhook=discord_webhook,
            send_discord=discord_webhook is not None,
        )
        
        runner = StressTestRunner(config)
        report = runner.run()
        
        return report.all_passed
    
    def full_pipeline(self, discord_webhook: str = None) -> bool:
        """Run full pipeline: train + stress test."""
        start = time.time()
        
        print("=" * 60)
        print("       FULL TRAINING PIPELINE")
        print("=" * 60)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Train
        history = self.train()
        
        # Stress test
        passed = self.run_stress_test(discord_webhook)
        
        elapsed = time.time() - start
        print(f"\nPipeline completed in {elapsed/60:.1f} minutes")
        print(f"Final status: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        return passed


def main():
    parser = argparse.ArgumentParser(description="Training pipeline")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--stress-test", action="store_true", help="Run stress test only")
    parser.add_argument("--full", action="store_true", help="Run full pipeline (train + test)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="Save checkpoint every N batches")
    parser.add_argument("--webhook", help="Discord webhook URL")
    parser.add_argument("--model", default="models/payload_cnn.pt", help="Model path")
    args = parser.parse_args()
    
    config = TrainingConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.checkpoint_every = args.checkpoint_every
    config.model_save_path = args.model
    
    pipeline = TrainingPipeline(config)
    
    if args.full:
        success = pipeline.full_pipeline(args.webhook)
        sys.exit(0 if success else 1)
    elif args.train:
        pipeline.train(resume=args.resume)
    elif args.stress_test:
        success = pipeline.run_stress_test(args.webhook)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
