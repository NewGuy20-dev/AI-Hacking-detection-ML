"""Train PayloadCNN model with robust fault-tolerant pipeline."""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
from tqdm import tqdm

from torch_models.payload_cnn import PayloadCNN
from torch_models.utils import setup_gpu, EarlyStopping, save_model
from data.streaming_dataset import BalancedStreamingDataset

# Robust training imports
from training.robust import (
    RobustTrainingConfig,
    RobustCheckpointManager,
    TrainingMonitor,
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
    
    def training_started(self, model_type: str, epochs: int, train_size: int, gpu_name: str = None):
        embed = {
            "title": "🚀 Training Started",
            "color": self.COLORS['info'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Model", "value": model_type.upper(), "inline": True},
                {"name": "Epochs", "value": str(epochs), "inline": True},
                {"name": "Train Size", "value": f"{train_size:,}", "inline": True},
                {"name": "GPU", "value": gpu_name or "CPU", "inline": True},
                {"name": "Mode", "value": "🛡️ Robust Training", "inline": True},
            ]
        }
        return self._send(embed)
    
    def epoch_completed(self, model_type: str, epoch: int, total_epochs: int,
                        train_loss: float, val_acc: float, elapsed: float, eta: float, bad_samples: int = 0):
        embed = {
            "title": f"📊 Epoch {epoch}/{total_epochs} Complete",
            "color": self.COLORS['info'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Model", "value": model_type.upper(), "inline": True},
                {"name": "Train Loss", "value": f"{train_loss:.4f}", "inline": True},
                {"name": "Val Acc", "value": f"{val_acc:.2%}", "inline": True},
                {"name": "Elapsed", "value": str(timedelta(seconds=int(elapsed))), "inline": True},
                {"name": "ETA", "value": str(timedelta(seconds=int(eta))), "inline": True},
                {"name": "Bad Samples", "value": str(bad_samples), "inline": True},
            ]
        }
        return self._send(embed)
    
    def training_finished(self, model_type: str, total_time: float, best_acc: float, epochs: int):
        embed = {
            "title": "✅ Training Complete!",
            "color": self.COLORS['success'],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fields": [
                {"name": "Model", "value": model_type.upper(), "inline": True},
                {"name": "Best Val Acc", "value": f"{best_acc:.2%}", "inline": True},
                {"name": "Total Time", "value": str(timedelta(seconds=int(total_time))), "inline": True},
                {"name": "Epochs", "value": str(epochs), "inline": True},
            ]
        }
        return self._send(embed)
    
    def training_error(self, model_type: str, error: str, epoch: int = None):
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


discord = DiscordNotifier()


def get_all_data_files(base_path):
    """Get all malicious and benign data files."""
    base = Path(base_path)
    
    # === MALICIOUS FILES ===
    malicious_files = []
    
    # security_payloads directory (txt files - need to convert)
    payloads_dir = base / 'datasets' / 'security_payloads'
    if payloads_dir.exists():
        for folder in ['injection', 'fuzzing', 'misc', 'PayloadsAllTheThings', 'SecLists']:
            folder_path = payloads_dir / folder
            if folder_path.exists():
                malicious_files.extend(folder_path.rglob('*.txt'))
                malicious_files.extend(folder_path.rglob('*.lst'))
                malicious_files.extend(folder_path.rglob('*.list'))
    
    # === BENIGN FILES ===
    benign_files = []
    
    # benign_60m directory (JSONL - ~60M samples)
    benign_60m = base / 'datasets' / 'benign_60m'
    if benign_60m.exists():
        benign_files.extend(benign_60m.glob('*.jsonl'))
    
    # live_benign directory (NEW - ~190M samples from live sources)
    live_benign = base / 'datasets' / 'live_benign'
    if live_benign.exists():
        benign_files.extend(live_benign.glob('*.jsonl'))
    
    # benign_5m.jsonl (~5M samples)
    benign_5m = base / 'datasets' / 'benign_5m.jsonl'
    if benign_5m.exists():
        benign_files.append(benign_5m)
    
    # fp_test_500k.jsonl (500K samples)
    fp_test = base / 'datasets' / 'fp_test_500k.jsonl'
    if fp_test.exists():
        benign_files.append(fp_test)
    
    # curated_benign directory
    curated_dir = base / 'datasets' / 'curated_benign'
    if curated_dir.exists():
        benign_files.extend(curated_dir.glob('*.txt'))
        benign_files.extend(curated_dir.glob('*.jsonl'))
        # adversarial subdirectory
        adv_dir = curated_dir / 'adversarial'
        if adv_dir.exists():
            benign_files.extend(adv_dir.glob('*.txt'))
    
    return malicious_files, benign_files


def count_file_lines(files):
    """Estimate total lines in files."""
    total = 0
    for f in files[:5]:  # Sample first 5 files
        try:
            with open(f, 'r', errors='ignore') as fp:
                total += sum(1 for _ in fp)
        except:
            pass
    if len(files) > 5:
        total = total * len(files) // 5
    return total


def worker_init_fn(worker_id):
    """Initialize worker with unique random seed."""
    import random
    import numpy as np
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train():
    """Main training function with streaming dataset and robust pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint-every', type=int, default=500, help='Save checkpoint every N batches')
    parser.add_argument('--debug', action='store_true', help='Debug mode (num_workers=0)')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--samples-per-epoch', type=int, default=20_000_000, help='Samples per epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    # Setup
    base_path = Path(__file__).parent.parent.parent
    device = setup_gpu()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    start_time = time.time()
    epoch = 0
    
    # Config
    config = RobustTrainingConfig(
        debug_mode=args.debug,
        checkpoint_dir=str(base_path / 'checkpoints' / 'payload'),
        checkpoint_every_n_batches=args.checkpoint_every,
        log_dir=str(base_path / 'logs'),
    )
    
    ckpt_mgr = RobustCheckpointManager.from_config(config, 'payload_cnn')
    
    try:
        # Get all data files
        print("\n--- Loading Data Files ---")
        malicious_files, benign_files = get_all_data_files(base_path)
        
        print(f"Malicious sources: {len(malicious_files)} files")
        print(f"Benign sources: {len(benign_files)} files")
        
        if not malicious_files:
            print("ERROR: No malicious data files found!")
            return
        if not benign_files:
            print("ERROR: No benign data files found!")
            return
        
        # Print file details
        print("\nBenign files:")
        for f in benign_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  - {f.name}: {size_mb:.1f} MB")
        
        # Model
        print("\n--- Creating Model ---")
        model = PayloadCNN(vocab_size=256, embed_dim=128, num_filters=256, max_len=500).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {param_count:,}")
        
        # Training setup
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scaler = GradScaler()
        
        batches_per_epoch = args.samples_per_epoch // args.batch_size
        total_steps = batches_per_epoch * args.epochs + 1  # +1 buffer for edge case
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, total_steps=total_steps,
            pct_start=0.1, anneal_strategy='cos'
        )
        
        # Resume
        start_epoch, start_batch, global_step = 0, 0, 0
        if args.resume:
            resume_info = ckpt_mgr.load(model, optimizer, scheduler, scaler, device)
            start_epoch = resume_info['epoch']
            start_batch = resume_info['batch_idx']
            global_step = resume_info['global_step']
            if start_batch >= batches_per_epoch:
                start_epoch += 1
                start_batch = 0
            print(f"Resuming from epoch {start_epoch}, batch {start_batch}")
        
        # Monitor
        monitor = TrainingMonitor(config)
        monitor.start()
        
        # 🚀 Discord notification
        discord.training_started('payload', args.epochs, args.samples_per_epoch, gpu_name)
        
        print(f"\n--- Training (Streaming Mode) ---")
        print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
        print(f"Samples/epoch: {args.samples_per_epoch:,}, Batches/epoch: {batches_per_epoch:,}")
        
        best_loss = float('inf')
        best_state = None
        
        for epoch in range(start_epoch, args.epochs):
            model.train()
            monitor.set_training_active(True)
            
            # Create streaming dataset (fresh each epoch for different shuffle)
            dataset = BalancedStreamingDataset(
                malicious_files, benign_files,
                max_len=500,
                samples_per_epoch=args.samples_per_epoch
            )
            
            num_workers = 0 if args.debug else min(4, 6)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
                worker_init_fn=worker_init_fn,
                prefetch_factor=4 if num_workers > 0 else None,
            )
            
            epoch_start_batch = start_batch if epoch == start_epoch else 0
            total_loss = 0
            batches_processed = 0
            
            pbar = tqdm(
                enumerate(loader),
                total=batches_per_epoch,
                initial=epoch_start_batch,
                desc=f"Epoch {epoch+1}/{args.epochs}"
            )
            
            for batch_idx, (inputs, targets) in pbar:
                if batch_idx < epoch_start_batch:
                    continue
                
                global_step += 1
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if scheduler._step_count < scheduler.total_steps:
                    scheduler.step()
                
                total_loss += loss.item()
                batches_processed += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
                
                # Checkpoint
                if ckpt_mgr.should_save(batch_idx):
                    ckpt_mgr.save(epoch, batch_idx, model, optimizer, scheduler, scaler, global_step)
            
            pbar.close()
            del loader
            torch.cuda.empty_cache()
            
            monitor.set_training_active(False)
            
            # End of epoch
            avg_loss = total_loss / max(batches_processed, 1)
            ckpt_mgr.save(epoch, batches_per_epoch, model, optimizer, scheduler, scaler, global_step)
            
            elapsed = time.time() - start_time
            eta = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1) if epoch >= start_epoch else 0
            
            print(f"\nEpoch {epoch+1}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
            print(f"Elapsed: {timedelta(seconds=int(elapsed))}, ETA: {timedelta(seconds=int(eta))}")
            
            # 📊 Discord
            discord.epoch_completed('payload', epoch+1, args.epochs, avg_loss, 0, elapsed, eta, 0)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_state = model.state_dict().copy()
            
            start_batch = 0  # Reset for next epoch
        
        monitor.stop()
        
        # Save model
        print("\n--- Saving Model ---")
        if best_state:
            model.load_state_dict(best_state)
        
        models_dir = base_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        model.eval()
        example = torch.zeros(1, 500, dtype=torch.long).to(device)
        save_model(model, models_dir / 'payload_cnn', example)
        torch.save(best_state or model.state_dict(), models_dir / 'payload_cnn.pth')
        
        total_time = time.time() - start_time
        print(f"✓ Model saved to models/payload_cnn.pt")
        print(f"✓ Best loss: {best_loss:.4f}")
        print(f"✓ Total time: {timedelta(seconds=int(total_time))}")
        
        # ✅ Discord
        discord.training_finished('payload', total_time, best_loss, args.epochs)
        
    except Exception as e:
        discord.training_error('payload', f"{type(e).__name__}: {str(e)}", epoch)
        raise


if __name__ == "__main__":
    train()
