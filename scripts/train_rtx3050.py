#!/usr/bin/env python3
"""
Complete training script for all 6 models with specialized data mapping.
Optimized for RTX 3050 + i5 12th Gen.
"""
import sys
import time
import argparse
import traceback
import json
import signal
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

import torch

# Graceful shutdown handling (for thermal guardian)
_shutdown_requested = False

def _handle_shutdown(signum, frame):
    """Handle SIGTERM/SIGINT - set flag for graceful shutdown."""
    global _shutdown_requested
    print("\n⚠️ Shutdown requested - will save checkpoint after current batch...")
    _shutdown_requested = True

signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm

# Sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.torch_models.payload_cnn import PayloadCNN
from src.torch_models.url_cnn import URLCNN
from src.torch_models.timeseries_lstm import TimeSeriesLSTM
from src.data.streaming_dataset import StreamingDataset, BalancedStreamingDataset
from src.training.checkpoint_manager import CheckpointManager
from src.training.transfer_learning import (
    freeze_embeddings, unfreeze_all, count_trainable_params,
    GradualUnfreezer, get_transfer_learning_schedule,
    create_discriminative_optimizer, WarmupScheduler,
    ProgressiveUnfreezer, freeze_all_except
)

# Data mapping for all 6 models
DATA_MAPPING = {
    'payload': {
        'num': 1,
        'name': 'PAYLOAD CNN',
        'total': '~286M samples',
        'malicious': '94.5M security payloads (SQL, XSS, command injection)',
        'benign': '192M curated + live benign',
        'desc': 'Character-level CNN for injection attack detection'
    },
    'url': {
        'num': 2,
        'name': 'URL CNN',
        'total': '16.2M samples',
        'malicious': '5.2M malicious URLs (phishing, malware)',
        'benign': '11M benign URLs (Tranco + Common Crawl)',
        'desc': 'Character-level CNN for URL classification'
    },
    'timeseries': {
        'num': 3,
        'name': 'TIMESERIES LSTM',
        'total': '1.5M samples',
        'malicious': '250K attack sequences (DDoS, portscan)',
        'benign': '1.25M normal traffic sequences',
        'desc': 'LSTM for temporal anomaly detection'
    },
    'network': {
        'num': 4,
        'name': 'NETWORK INTRUSION RF',
        'total': '6.5M samples',
        'malicious': '5.5M attack samples (DoS, Probe, R2L, U2R)',
        'benign': '1M MAWI normal traffic',
        'desc': 'RandomForest for 41-feature network traffic classification'
    },
    'fraud': {
        'num': 5,
        'name': 'FRAUD DETECTION XGBOOST',
        'total': '5.8M samples',
        'malicious': '785K fraudulent transactions',
        'benign': '5M normal transactions',
        'desc': 'XGBoost for transaction pattern classification'
    },
    'host': {
        'num': 6,
        'name': 'HOST BEHAVIOR RF',
        'total': '5.5M samples',
        'malicious': '500K malware patterns',
        'benign': '5M normal host behavior',
        'desc': 'RandomForest for process/memory pattern classification'
    }
}

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


class EarlyStopping:
    """Stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
    
    def check(self, val_loss: float, epoch: int) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"⏹️ Early stopping: val_loss hasn't improved for {self.patience} epochs")
            print(f"   Best: {self.best_loss:.4f} at epoch {self.best_epoch + 1}")
            return True
        return False


def get_url_data_splits(config: TrainingConfig):
    """Get separate train/val file splits for URL model."""
    url_dir = config.data_dir / 'url_analysis'
    
    # Training files
    train_mal = [f for f in [
        url_dir / 'urlhaus.csv',
        url_dir / 'kaggle_malicious_urls.csv',
        url_dir / 'malicious_urls_5m.jsonl',
    ] if f.exists()]
    
    train_ben = [f for f in [
        url_dir / 'top-1m.csv',
        config.data_dir / 'live_benign' / 'common_crawl_urls.jsonl',
    ] if f.exists()]
    
    # Validation files (held-out)
    val_mal = [f for f in [
        url_dir / 'malicious_urls' / 'malicious_phish.csv',
        url_dir / 'synthetic_malicious_hard.txt',
    ] if f.exists()]
    
    val_ben = [f for f in [
        url_dir / 'synthetic_benign_hard.txt',
    ] if f.exists()]
    
    return (train_mal, train_ben), (val_mal, val_ben)


def validate_epoch(model, val_loader, config) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)
            
            if config.use_amp:
                with autocast('cuda'):
                    logits = model(x)
                    loss = F.binary_cross_entropy_with_logits(logits, y)
            else:
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / max(num_batches, 1)


def get_data_files(config: TrainingConfig, model_type: str):
    """Get data files for model type."""
    malicious_files = []
    benign_files = []
    
    if model_type == 'url':
        # URL-specific data: Only actual URLs
        url_dir = config.data_dir / 'live_benign'
        benign_url_file = url_dir / 'common_crawl_urls.jsonl'
        if benign_url_file.exists():
            benign_files.append(benign_url_file)
        
        # Malicious URLs
        malicious_url_dir = config.data_dir / 'malicious_urls'
        if malicious_url_dir.exists():
            malicious_files.extend(malicious_url_dir.rglob('*.jsonl'))
        
        # Fallback: If no malicious URLs, generate them
        if not malicious_files:
            print("⚠️  No malicious URLs found. Run: python scripts/generate_malicious_urls.py")
            print("⚠️  Using security payloads as fallback (not ideal for URL training)")
            payload_dir = config.data_dir / 'security_payloads'
            if payload_dir.exists():
                malicious_files.extend(list(payload_dir.rglob('*.txt'))[:100])  # Limit to 100 files
    else:
        # Payload/other models: Use security payloads
        payload_dir = config.data_dir / 'security_payloads'
        if payload_dir.exists():
            malicious_files.extend(payload_dir.rglob('*.txt'))
            malicious_files.extend(payload_dir.rglob('*.jsonl'))
        
        # Benign sources
        benign_dirs = [
            config.data_dir / 'benign_60m',
            config.data_dir / 'curated_benign',
            config.data_dir / 'live_benign',
        ]
        for d in benign_dirs:
            if d.exists():
                benign_files.extend(d.rglob('*.jsonl'))
                benign_files.extend(d.rglob('*.txt'))
        
        # Additional benign
        for f in ['benign_5m.jsonl', 'fp_test_500k.jsonl']:
            p = config.data_dir / f
            if p.exists():
                benign_files.append(p)
    
    return malicious_files, benign_files


def create_model(model_type: str, device: str) -> nn.Module:
    """Create model instance with correct parameters."""
    if model_type == 'payload':
        model = PayloadCNN(vocab_size=256, embed_dim=128, max_len=500)
    elif model_type == 'url':
        # URLCNN only takes vocab_size, embed_dim, max_len (no num_filters)
        model = URLCNN(vocab_size=128, embed_dim=64, max_len=200)
    elif model_type == 'timeseries':
        # TimeSeriesLSTM takes input_dim, hidden_dim, num_layers (not input_size)
        model = TimeSeriesLSTM(input_dim=10, hidden_dim=128, num_layers=2)
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
        # Check for graceful shutdown (thermal guardian or Ctrl+C)
        global _shutdown_requested
        if _shutdown_requested:
            print("\n💾 Saving checkpoint before shutdown...")
            checkpoint_mgr.save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, batch_idx, total_loss / max(batch_count, 1), model_name
            )
            print(f"✅ Checkpoint saved at epoch {epoch+1}, batch {batch_idx}")
            print(f"   Resume with: python scripts/train_rtx3050.py --model {model_name} --resume")
            sys.exit(0)
        
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
        
        # Step scheduler AFTER optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        batch_count += 1
        avg_loss = total_loss / batch_count
        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config.lr
        
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
    """Train a single model with transfer learning, validation, and progressive unfreezing."""
    
    # Check if already completed
    final_model_path = Path(config.model_dir) / f"{model_type}_cnn.pth"
    if final_model_path.exists() and not (hasattr(config, 'resume') and config.resume):
        print(f"\n✓ {model_type.upper()} already trained, skipping...")
        return
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Samples per epoch: {config.samples_per_epoch:,}")
    print(f"Progressive unfreezing: FC(0-1) → Conv(2) → Embed(3+)")
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
        
        # Setup progressive unfreezing (FC → Conv → Embed)
        unfreezer = ProgressiveUnfreezer(model, config.lr)
        
        # Initialize: freeze all except FC
        from src.training.transfer_learning import freeze_all_except
        freeze_all_except(model, ['fc', 'linear'])
        trainable = count_trainable_params(model)
        print(f"Stage 1: FC only, trainable params: {trainable:,}")
        
        # Initial optimizer (FC only)
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr, weight_decay=config.weight_decay
        )
        
        # Get data files
        if model_type == 'url':
            # Use separate train/val files for URL model
            (train_mal, train_ben), (val_mal, val_ben) = get_url_data_splits(config)
            malicious_files, benign_files = train_mal, train_ben
            print(f"Train - Malicious: {len(train_mal)}, Benign: {len(train_ben)}")
            print(f"Val - Malicious: {len(val_mal)}, Benign: {len(val_ben)}")
            
            # Import RealURLDataset for URL training
            from src.data.url_dataset import RealURLDataset
            use_real_url_dataset = True
        else:
            malicious_files, benign_files = get_data_files(config, model_type)
            val_mal, val_ben = malicious_files, benign_files  # Same files for non-URL
            use_real_url_dataset = False
            print(f"Malicious files: {len(malicious_files)}")
            print(f"Benign files: {len(benign_files)}")
        
        if not malicious_files and not benign_files:
            print("No data files found!")
            return
        
        # Estimate batches
        batches_per_epoch = config.samples_per_epoch // config.batch_size
        val_samples = config.samples_per_epoch // 10  # 10% for validation
        
        # Scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=batches_per_epoch, T_mult=1, eta_min=config.lr * 0.01
        )
        
        # Scaler for AMP
        scaler = GradScaler(device="cuda") if config.use_amp else None
        
        # Checkpoint manager
        checkpoint_mgr = CheckpointManager(
            config.checkpoint_dir, config.model_dir, keep_last=3
        )
        
        # Early stopping
        early_stopper = EarlyStopping(patience=3, min_delta=0.001)
        
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
                print(f"Resuming from epoch {start_epoch}, batch {start_batch}")
                if start_batch >= batches_per_epoch:
                    start_epoch += 1
                    start_batch = 0
            else:
                print("No checkpoint found, starting fresh")
        else:
            checkpoint_mgr.clear_all_checkpoints()
        
        # Training loop
        best_loss = float('inf')
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, config.epochs):
            epoch_start_batch = start_batch if epoch == start_epoch else 0
            
            # Check progressive unfreezing
            changed, lr_mults = unfreezer.step(epoch)
            if changed and lr_mults:
                optimizer = unfreezer.create_optimizer(lr_mults, config.weight_decay)
                base_scheduler = CosineAnnealingWarmRestarts(
                    optimizer, T_0=batches_per_epoch, T_mult=1, eta_min=config.lr * 0.001
                )
                warmup_steps = int(batches_per_epoch * 0.1)
                scheduler = WarmupScheduler(optimizer, warmup_steps, base_scheduler)
            
            # Create training dataset
            if use_real_url_dataset:
                dataset = RealURLDataset(
                    malicious_files, benign_files,
                    max_len=200, samples_per_epoch=config.samples_per_epoch, vocab_size=128
                )
            else:
                dataset = BalancedStreamingDataset(
                    malicious_files, benign_files,
                    max_len=500 if model_type == 'payload' else 200,
                    samples_per_epoch=config.samples_per_epoch,
                    vocab_size=128 if model_type == 'url' else 256
                )
            
            import platform
            is_windows = platform.system() == 'Windows'
            
            loader = DataLoader(
                dataset, batch_size=config.batch_size, num_workers=0,
                pin_memory=config.pin_memory,
                worker_init_fn=worker_init_fn if not is_windows else None,
            )
            
            # Train epoch
            epoch_loss = train_epoch(
                model, loader, optimizer, scheduler, scaler, config,
                epoch, checkpoint_mgr, model_type, batches_per_epoch, epoch_start_batch
            )
            
            del loader
            torch.cuda.empty_cache()
            
            # Validation
            if model_type == 'url' and val_mal and val_ben:
                val_dataset = RealURLDataset(
                    val_mal, val_ben, max_len=200,
                    samples_per_epoch=val_samples, vocab_size=128
                )
            else:
                val_dataset = BalancedStreamingDataset(
                    val_mal, val_ben,
                    max_len=500 if model_type == 'payload' else 200,
                    samples_per_epoch=val_samples,
                    vocab_size=128 if model_type == 'url' else 256
                )
            
            val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
            val_loss = validate_epoch(model, val_loader, config)
            del val_loader
            
            # Track best
            if epoch_loss < best_loss:
                best_loss = epoch_loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_mgr.save_final_model(model, f"{model_type}_cnn_best")
            
            # Progress
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (config.epochs - epoch - 1)
            print(f"\nEpoch {epoch+1}: train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}")
            print(f"Elapsed: {timedelta(seconds=int(elapsed))}, ETA: {timedelta(seconds=int(eta))}")
            
            # 📊 Discord: Epoch completed
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else config.lr
            discord.epoch_completed(model_type, epoch+1, config.epochs, epoch_loss, 
                                   current_lr, elapsed, eta)
            
            # Early stopping check
            if early_stopper.check(val_loss, epoch):
                print(f"🛑 Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        checkpoint_mgr.save_final_model(model, f"{model_type}_cnn")
        
        # Delete all checkpoints after training
        checkpoint_mgr.cleanup_all_after_training()
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"{model_type.upper()} training complete!")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Best train_loss: {best_loss:.4f}")
        print(f"Best val_loss: {best_val_loss:.4f}")
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


def print_data_mapping(model_type: str):
    """Print specialized data mapping for a model."""
    if model_type not in DATA_MAPPING:
        return
    
    m = DATA_MAPPING[model_type]
    print(f"\n[{m['num']}/6] {m['name']}")
    print("-" * 80)
    print(f"  Description: {m['desc']}")
    print(f"  Malicious: {m['malicious']}")
    print(f"  Benign: {m['benign']}")
    print(f"  Total: {m['total']}")
    print("-" * 80)


def train_sklearn_model(model_type: str, config: TrainingConfig):
    """Train sklearn models (network, fraud, host) with checkpoint support."""
    print_data_mapping(model_type)
    
    # Check if already completed
    checkpoint_file = config.checkpoint_dir / f"{model_type}_sklearn_complete.json"
    if checkpoint_file.exists():
        print(f"  ✓ {model_type.upper()} already trained (checkpoint found)")
        return True
    
    # Discord: Started
    discord.training_started(model_type, config, "CPU (sklearn)")
    
    start_time = time.time()
    base_dir = config.data_dir
    
    try:
        print(f"\n⚠️  Sklearn model training:")
        print(f"  This will call the existing training script for {model_type}")
        print(f"  The script will handle data loading and model training")
        
        # Determine which script to call
        base_path = Path(__file__).parent.parent  # Project root
        scripts_map = {
            'network': base_path / 'src' / 'train_network_intrusion.py',
            'fraud': base_path / 'src' / 'train_fraud_detection.py',
            'host': base_path / 'src' / 'train_host_behavior.py'
        }
        
        script_path = scripts_map.get(model_type)
        if not script_path or not script_path.exists():
            print(f"  ✗ Training script not found: {script_path}")
            return False
        
        # Call the training script using the current Python interpreter (venv)
        import subprocess
        import os
        
        # Use sys.executable to ensure we use the same Python (venv) that's running this script
        python_exe = sys.executable
        
        # Set environment to inherit current venv
        env = os.environ.copy()
        
        print(f"  Calling: {python_exe} {script_path.name}")
        
        result = subprocess.run(
            [python_exe, str(script_path)],
            cwd=script_path.parent,
            env=env
        )
        
        success = result.returncode == 0
        elapsed = time.time() - start_time
        
        if success:
            # Save completion checkpoint
            config.checkpoint_dir.mkdir(exist_ok=True)
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'model': model_type,
                    'completed': True,
                    'timestamp': datetime.now().isoformat(),
                    'elapsed_seconds': elapsed
                }, f, indent=2)
            print(f"  ✓ {model_type.upper()} training completed in {timedelta(seconds=int(elapsed))}")
        else:
            print(f"  ✗ {model_type.upper()} training failed")
        
        # Discord: Completed
        discord.training_finished(model_type, elapsed, 0.0, 1, 0)
        
        return success
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        discord.training_error(model_type, error_msg)
        print(f"  ✗ Error: {error_msg}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train all 6 models with specialized data')
    parser.add_argument('--model', type=str, default='all',
                       choices=['payload', 'url', 'timeseries', 'network', 'fraud', 'host', 
                               'pytorch', 'sklearn', 'all'])
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
    
    # Print dataset overview
    print("=" * 80)
    print(" COMPLETE MODEL RETRAINING PIPELINE")
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\n📊 DATASET OVERVIEW")
    print("  Total: 322.6M samples (54.51 GB)")
    print("  Benign: 215.6M (66.8%)")
    print("  Malicious: 107.1M (33.2%)")
    print("  Ratio: 2:1 (optimal for production)")
    print("=" * 80)
    
    # Print GPU info
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu} ({mem:.1f} GB)")
    else:
        print("\nWARNING: No GPU, using CPU")
    
    start = time.time()
    
    # Determine which models to train
    pytorch_models = ['payload', 'url']  # Skip timeseries - needs different data format
    sklearn_models = ['network', 'fraud', 'host']
    
    if args.model == 'all':
        models_to_train = pytorch_models + sklearn_models
    elif args.model == 'pytorch':
        models_to_train = pytorch_models
    elif args.model == 'sklearn':
        models_to_train = sklearn_models
    else:
        models_to_train = [args.model]
    
    # Train models
    models_trained = []
    results = {}
    
    for model_type in models_to_train:
        try:
            if model_type in pytorch_models:
                print_data_mapping(model_type)
                train_model(model_type, config)
                results[model_type] = True
            else:
                success = train_sklearn_model(model_type, config)
                results[model_type] = success if success is not None else False
            models_trained.append(model_type)
        except Exception as e:
            print(f"\n✗ {model_type.upper()} training failed: {e}")
            results[model_type] = False
    
    total = time.time() - start
    successful = sum(1 for v in results.values() if v)
    failed = len(results) - successful
    
    print(f"\n{'='*80}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Total time: {timedelta(seconds=int(total))}")
    print(f"Successful: {successful}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print("\nModel Status:")
    for model_type, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        name = DATA_MAPPING.get(model_type, {}).get('name', model_type.upper())
        print(f"  {name:30s}: {status}")
    print(f"{'='*80}")
    
    # 🎉 Discord: All training complete
    if len(models_trained) > 1:
        discord.all_training_complete(total, models_trained)


if __name__ == '__main__':
    main()
