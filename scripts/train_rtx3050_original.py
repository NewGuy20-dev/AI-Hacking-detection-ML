#!/usr/bin/env python3
"""
RTX 3050 optimized training script with transfer learning.
Target: 5-6 hours total training time.
"""
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.torch_models.payload_cnn import PayloadCNN
from src.torch_models.url_cnn import URLCNN
from src.torch_models.timeseries_lstm import TimeSeriesLSTM
from src.data.streaming_dataset import StreamingDataset, BalancedStreamingDataset
from src.training.checkpoint_manager import CheckpointManager
from src.training.transfer_learning import (
    freeze_embeddings, unfreeze_all, count_trainable_params,
    GradualUnfreezer, get_transfer_learning_schedule
)

# Discord notifications
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1452715933398466782/Ajftu5_fHelFqifTRcZN3S7fCDddXPs89p9w8dTHX8pF1xUO59ckac_DyCTQsRKC1H8O"


class DiscordNotifier:
    """Send training notifications to Discord."""
    
    COLORS = {'info': 0x3498db, 'success': 0x2ecc71, 'warning': 0xf39c12, 'error': 0xe74c3c}
    
    def __init__(self, webhook_url: str = DISCORD_WEBHOOK):
        self.webhook_url = webhook_url
        self.enabled = HAS_REQUESTS and webhook_url
    
    def _send(self, embed: dict) -> bool:
        if not self.enabled:
            return False
        try:
            requests.post(self.webhook_url, json={"embeds": [embed]}, timeout=10)
            return True
        except:
            return False
    
    def training_started(self, model_type: str, config, gpu_name: str = None):
        """🚀 Training started notification."""
        embed = {
            "title": "🚀 Training Started",
            "color": self.COLORS['info'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Model", "value": model_type.upper(), "inline": True},
                {"name": "Epochs", "value": str(config.epochs), "inline": True},
                {"name": "Batch Size", "value": str(config.batch_size), "inline": True},
                {"name": "Samples/Epoch", "value": f"{config.samples_per_epoch:,}", "inline": True},
                {"name": "Freeze Epochs", "value": str(config.freeze_epochs), "inline": True},
                {"name": "GPU", "value": gpu_name or "CPU", "inline": True},
            ]
        }
        return self._send(embed)
    
    def epoch_completed(self, model_type: str, epoch: int, total_epochs: int, 
                        loss: float, lr: float, elapsed: float, eta: float):
        """📊 Epoch completed notification."""
        embed = {
            "title": f"📊 Epoch {epoch}/{total_epochs} Complete",
            "color": self.COLORS['info'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Model", "value": model_type.upper(), "inline": True},
                {"name": "Loss", "value": f"{loss:.4f}", "inline": True},
                {"name": "LR", "value": f"{lr:.2e}", "inline": True},
                {"name": "Elapsed", "value": str(timedelta(seconds=int(elapsed))), "inline": True},
                {"name": "ETA", "value": str(timedelta(seconds=int(eta))), "inline": True},
            ]
        }
        return self._send(embed)
    
    def training_finished(self, model_type: str, total_time: float, best_loss: float, 
                          epochs: int, samples_trained: int):
        """✅ Training finished notification."""
        embed = {
            "title": "✅ Training Complete!",
            "color": self.COLORS['success'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Model", "value": model_type.upper(), "inline": True},
                {"name": "Best Loss", "value": f"{best_loss:.4f}", "inline": True},
                {"name": "Total Time", "value": str(timedelta(seconds=int(total_time))), "inline": True},
                {"name": "Epochs", "value": str(epochs), "inline": True},
                {"name": "Samples Trained", "value": f"{samples_trained:,}", "inline": True},
            ]
        }
        return self._send(embed)
    
    def training_error(self, model_type: str, error: str, epoch: int = None):
        """❌ Training error notification."""
        embed = {
            "title": "❌ Training Error!",
            "color": self.COLORS['error'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Model", "value": model_type.upper(), "inline": True},
                {"name": "Epoch", "value": str(epoch) if epoch else "N/A", "inline": True},
                {"name": "Error", "value": f"```{error[:500]}```", "inline": False},
            ]
        }
        return self._send(embed)
    
    def all_training_complete(self, total_time: float, models_trained: list):
        """🎉 All training complete notification."""
        embed = {
            "title": "🎉 All Training Complete!",
            "color": self.COLORS['success'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Models", "value": ", ".join(m.upper() for m in models_trained), "inline": True},
                {"name": "Total Time", "value": str(timedelta(seconds=int(total_time))), "inline": True},
            ]
        }
        return self._send(embed)


# Global notifier
discord = DiscordNotifier()


def worker_init_fn(worker_id):
    """Initialize worker with unique random seed to prevent collisions."""
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class TrainingConfig:
    """Training configuration for Intel i5 12th Gen + RTX 3050."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1024
    num_workers = 6  # 6 workers for i5 12th Gen (6 P-cores)
    pin_memory = True
    prefetch_factor = 4  # Prefetch 4 batches per worker
    use_amp = True
    
    # Training params
    epochs = 5
    lr = 1e-3
    weight_decay = 1e-5
    gradient_clip = 1.0
    freeze_epochs = 2  # Epochs with frozen embeddings
    
    # Sampling
    samples_per_epoch = 20_000_000  # 20M per epoch
    
    # Checkpointing
    checkpoint_every = 1000
    val_every = 5000
    
    # Paths
    data_dir = Path('datasets')
    model_dir = Path('models')
    checkpoint_dir = Path('checkpoints')


def get_data_files(config: TrainingConfig, model_type: str):
    """Get data files for model type."""
    malicious_files = []
    benign_files = []
    
    # Malicious sources
    payload_dir = config.data_dir / 'security_payloads'
    if payload_dir.exists():
        malicious_files.extend(payload_dir.rglob('*.jsonl'))
    
    # Benign sources
    benign_dirs = [
        config.data_dir / 'benign_60m',
        config.data_dir / 'curated_benign',
    ]
    for d in benign_dirs:
        if d.exists():
            benign_files.extend(d.rglob('*.jsonl'))
    
    # Additional benign
    for f in ['benign_5m.jsonl', 'fp_test_500k.jsonl']:
        p = config.data_dir / f
        if p.exists():
            benign_files.append(p)
    
    return malicious_files, benign_files


def create_model(model_type: str, device: str) -> nn.Module:
    """Create model instance."""
    if model_type == 'payload':
        model = PayloadCNN(vocab_size=256, embed_dim=128, num_filters=256, max_len=500)
    elif model_type == 'url':
        model = URLCNN(vocab_size=128, embed_dim=64, num_filters=128, max_len=200)
    elif model_type == 'timeseries':
        model = TimeSeriesLSTM(input_size=10, hidden_size=128, num_layers=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_epoch(model, loader, optimizer, scheduler, scaler, config, 
                epoch, checkpoint_mgr, model_name, total_iters, start_batch=0, global_step=0):
    """Train for one epoch with tqdm progress bar and mid-epoch resume."""
    model.train()
    total_loss = 0
    batch_count = 0
    
    pbar = tqdm(enumerate(loader), total=total_iters, initial=start_batch,
                desc=f"Epoch {epoch+1}", unit="batch", dynamic_ncols=True)
    
    for batch_idx, (x, y) in pbar:
        # Skip already-processed batches on resume
        if batch_idx < start_batch:
            continue
        
        global_step += 1
        # Move to GPU (non-blocking with pinned memory)
        x = x.to(config.device, non_blocking=True)
        y = y.to(config.device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if config.use_amp:
            with autocast('cuda'):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
        
        scheduler.step()
        
        total_loss += loss.item()
        batch_count += 1
        avg_loss = total_loss / batch_count
        lr = scheduler.get_last_lr()[0]
        
        # Update tqdm with metrics
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{lr:.2e}',
            'samples/s': f'{pbar.format_dict["rate"] * config.batch_size:.0f}' if pbar.format_dict.get("rate") else '...'
        })
        
        # Checkpoint
        if batch_idx > 0 and batch_idx % config.checkpoint_every == 0:
            checkpoint_mgr.save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, batch_idx, avg_loss, model_name
            )
    
    # End-of-epoch checkpoint
    checkpoint_mgr.save_checkpoint(
        model, optimizer, scheduler, scaler,
        epoch, total_iters, total_loss / max(batch_count, 1), model_name
    )
    
    return total_loss / max(batch_count, 1)


def train_model(model_type: str, config: TrainingConfig):
    """Train a single model with transfer learning."""
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Samples per epoch: {config.samples_per_epoch:,}")
    print(f"Transfer learning: Freeze epochs 0-{config.freeze_epochs-1}")
    print(f"{'='*60}\n")
    
    # Get GPU name
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    
    # 🚀 Discord: Training started
    discord.training_started(model_type, config, gpu_name)
    
    epoch = 0  # Track for error reporting
    start_time = time.time()
    
    try:
        # Create model
        model = create_model(model_type, config.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Setup transfer learning
        freeze_embeddings(model)
        trainable = count_trainable_params(model)
        print(f"Trainable (frozen embeddings): {trainable:,}")
        
        unfreezer = GradualUnfreezer(
            model, 
            get_transfer_learning_schedule(config.epochs, config.freeze_epochs)
        )
        
        # Optimizer
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        
        # Get data
        malicious_files, benign_files = get_data_files(config, model_type)
        print(f"Malicious files: {len(malicious_files)}")
        print(f"Benign files: {len(benign_files)}")
        
        if not malicious_files and not benign_files:
            print("No data files found!")
            return
        
        # Estimate batches
        batches_per_epoch = config.samples_per_epoch // config.batch_size
        total_steps = batches_per_epoch * config.epochs
        
        # Scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Scaler for AMP (new PyTorch 3.12+ syntax)
        scaler = GradScaler(device="cuda") if config.use_amp else None
        
        # Checkpoint manager
        checkpoint_mgr = CheckpointManager(
            config.checkpoint_dir, config.model_dir, keep_last=3
        )
        
        # Resume from checkpoint or start fresh
        start_epoch, start_batch = 0, 0
        if hasattr(config, 'resume') and config.resume:
            latest_ckpt = checkpoint_mgr.get_latest_checkpoint(model_type)
            if latest_ckpt:
                ckpt_data = checkpoint_mgr.load_checkpoint(
                    latest_ckpt, model, optimizer, scheduler, scaler
                )
                start_epoch = ckpt_data['epoch']
                start_batch = ckpt_data['batch']
                print(f"Resuming training from epoch {start_epoch}, batch {start_batch}")
                # If we completed the epoch, move to next
                if start_batch >= batches_per_epoch:
                    start_epoch += 1
                    start_batch = 0
            else:
                print("No checkpoint found, starting fresh")
        else:
            # Clear old checkpoints before fresh training
            checkpoint_mgr.clear_all_checkpoints()
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(start_epoch, config.epochs):
            # Determine start batch for this epoch
            epoch_start_batch = start_batch if epoch == start_epoch else 0
            # Check if we should unfreeze
            if unfreezer.step(epoch):
                trainable = count_trainable_params(model)
                print(f"Trainable params now: {trainable:,}")
                
                # Recreate optimizer with all params
                optimizer = AdamW(model.parameters(), lr=config.lr / 2, 
                                weight_decay=config.weight_decay)
                remaining_steps = (config.epochs - epoch) * batches_per_epoch
                scheduler = OneCycleLR(
                    optimizer, max_lr=config.lr / 2,
                    total_steps=remaining_steps, pct_start=0.05
                )
            
            # Create fresh dataset each epoch (different shuffle)
            dataset = BalancedStreamingDataset(
                malicious_files, benign_files,
                max_len=500 if model_type == 'payload' else 200,
                samples_per_epoch=config.samples_per_epoch
            )
            
            # DataLoader optimized for i5 12th Gen + RTX 3050
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=min(config.num_workers, 4),
                pin_memory=config.pin_memory,
                persistent_workers=True,
                timeout=0,
                worker_init_fn=worker_init_fn,
                prefetch_factor=config.prefetch_factor,
            )
            
            # Train epoch
            epoch_loss = train_epoch(
                model, loader, optimizer, scheduler, scaler, config,
                epoch, checkpoint_mgr, model_type, batches_per_epoch, epoch_start_batch
            )
            
            # Explicitly delete loader to terminate workers on Windows
            del loader
            torch.cuda.empty_cache()
            
            # Track best
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                checkpoint_mgr.save_final_model(model, f"{model_type}_cnn_best")
            
            # Progress
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (config.epochs - epoch - 1)
            print(f"\nEpoch {epoch+1} complete: loss={epoch_loss:.4f}")
            print(f"Elapsed: {timedelta(seconds=int(elapsed))}, ETA: {timedelta(seconds=int(eta))}")
            print(f"Epoch {epoch+1} finished.")
            
            # 📊 Discord: Epoch completed
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config.lr
            discord.epoch_completed(model_type, epoch+1, config.epochs, epoch_loss, 
                                   current_lr, elapsed, eta)
        
        # Save final model
        checkpoint_mgr.save_final_model(model, f"{model_type}_cnn")
        
        # Delete all checkpoints after training
        checkpoint_mgr.cleanup_all_after_training()
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"{model_type.upper()} training complete!")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Best loss: {best_loss:.4f}")
        print(f"{'='*60}")
        
        # ✅ Discord: Training finished
        samples_trained = config.samples_per_epoch * config.epochs
        discord.training_finished(model_type, total_time, best_loss, config.epochs, samples_trained)
        
        return model
        
    except Exception as e:
        # ❌ Discord: Training error
        error_msg = f"{type(e).__name__}: {str(e)}"
        discord.training_error(model_type, error_msg, epoch)
        raise


def main():
    parser = argparse.ArgumentParser(description='Train models on RTX 3050')
    parser.add_argument('--model', type=str, default='all',
                       choices=['payload', 'url', 'timeseries', 'all'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--samples-per-epoch', type=int, default=20_000_000)
    parser.add_argument('--freeze-epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Update config
    config = TrainingConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.samples_per_epoch = args.samples_per_epoch
    config.freeze_epochs = args.freeze_epochs
    config.lr = args.lr
    config.resume = args.resume
    
    # Print GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu} ({mem:.1f} GB)")
    else:
        print("WARNING: No GPU, using CPU")
    
    start = time.time()
    
    # Train models
    models_trained = []
    if args.model == 'all':
        for m in ['payload', 'url', 'timeseries']:
            train_model(m, config)
            models_trained.append(m)
    else:
        train_model(args.model, config)
        models_trained.append(args.model)
    
    total = time.time() - start
    print(f"\n{'='*60}")
    print(f"ALL TRAINING COMPLETE")
    print(f"Total time: {timedelta(seconds=int(total))}")
    print(f"{'='*60}")
    
    # 🎉 Discord: All training complete
    if len(models_trained) > 1:
        discord.all_training_complete(total, models_trained)


if __name__ == '__main__':
    main()
