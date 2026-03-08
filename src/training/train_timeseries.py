"""Train Time-Series LSTM for network traffic anomaly detection."""
import sys
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler
from tqdm import tqdm

from src.torch_models.timeseries_lstm import TimeSeriesLSTM
from src.torch_models.datasets import TimeSeriesDataset
from src.torch_models.utils import setup_gpu, EarlyStopping, save_model
from src.training.checkpoint import CheckpointManager
from src.training.training_utils import (
    binary_metrics,
    load_operational_threshold,
    stratified_index_split,
    write_training_manifest,
)
from src.data_guardrails import assert_allowed_training_paths
from src.timeseries_synthetic import generate_stress_aligned_benign_sequences, generate_stress_aligned_normal_sequence
	

def _new_source_bucket() -> Dict[str, int]:
    return {"total": 0, "malicious": 0, "benign": 0}


def _bump_source(summary: Dict[str, Dict[str, int]], source: str, label: int, count: int = 1) -> None:
    bucket = summary.setdefault(source, _new_source_bucket())
    bucket["total"] += int(count)
    if int(label) == 1:
        bucket["malicious"] += int(count)
    else:
        bucket["benign"] += int(count)


def _summarize_sequences(sequences: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(sequences, dtype=np.float32)
    if arr.size == 0:
        return {
            "shape": list(arr.shape),
            "global": {},
            "per_feature": [],
        }

    flat = arr.reshape(-1)
    feature_stats = []
    for feat_idx in range(arr.shape[-1]):
        feat = arr[:, :, feat_idx].reshape(-1)
        feature_stats.append(
            {
                "feature": int(feat_idx),
                "min": float(feat.min()),
                "p50": float(np.percentile(feat, 50)),
                "p95": float(np.percentile(feat, 95)),
                "max": float(feat.max()),
                "mean": float(feat.mean()),
                "std": float(feat.std()),
            }
        )

    return {
        "shape": list(arr.shape),
        "global": {
            "min": float(flat.min()),
            "p50": float(np.percentile(flat, 50)),
            "p95": float(np.percentile(flat, 95)),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
        },
        "per_feature": feature_stats,
    }


def _append_source(
    samples: list[np.ndarray],
    labels: list[np.ndarray],
    source_names: list[np.ndarray],
    source_stats: Dict[str, Dict[str, Any]],
    source_counts: Dict[str, Dict[str, int]],
    source_name: str,
    data: np.ndarray,
    label: int,
) -> None:
    if data is None or len(data) == 0:
        return
    arr = np.asarray(data, dtype=np.float32)
    samples.append(arr)
    labels.append(np.full(arr.shape[0], int(label), dtype=np.float32))
    source_names.append(np.array([source_name] * arr.shape[0], dtype=object))
    _bump_source(source_counts, source_name, label, arr.shape[0])
    source_stats[source_name] = _summarize_sequences(arr)


def _ensure_timeseries_shape(data: np.ndarray, source_name: str) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (60, 8):
        raise ValueError(f"{source_name} has invalid shape {arr.shape}; expected [N, 60, 8]")
    return arr


def generate_benign_hard_negative_traffic(n_samples=5000, seq_len=60, n_features=8):
    """Generate benign sequences with bursts and spikes that should remain non-malicious."""
    sequences = []
    for _ in range(n_samples):
        base = generate_normal_traffic(1, seq_len=seq_len, n_features=n_features)[0]
        # Add a few short benign bursts without introducing attack-wide sustained patterns.
        for _burst in range(np.random.randint(1, 4)):
            start = np.random.randint(5, seq_len - 6)
            duration = np.random.randint(2, 6)
            base[start:start + duration, 0] *= np.random.uniform(1.1, 1.4)  # packets
            base[start:start + duration, 1] *= np.random.uniform(1.1, 1.5)  # bytes
            base[start:start + duration, 6] += np.random.uniform(0.03, 0.10)  # syn ratio
        base[:, 7] = np.clip(base[:, 7], 0.0, None)
        sequences.append(base.astype(np.float32))
    return np.array(sequences, dtype=np.float32)


def generate_stress_aligned_hard_negative_traffic(n_samples=10000):
    """Generate harder benign sequences that resemble stress benign traffic."""
    sequences = []
    for _ in range(n_samples):
        base = generate_stress_aligned_normal_sequence().copy()
        for _burst in range(np.random.randint(1, 4)):
            start = np.random.randint(5, 50)
            duration = np.random.randint(2, 8)
            base[start:start + duration, 0] *= np.random.uniform(1.05, 1.35)
            base[start:start + duration, 1] *= np.random.uniform(1.15, 1.80)
            base[start:start + duration, 2] += np.random.uniform(5.0, 25.0)
            base[start:start + duration, 3] = np.clip(
                base[start:start + duration, 3] + np.random.uniform(0.02, 0.10),
                0.0,
                0.35,
            )
        if np.random.random() < 0.4:
            pulse_idx = np.random.randint(8, 52)
            base[pulse_idx:pulse_idx + 4, 1] *= np.random.uniform(2.0, 4.5)
        base = np.clip(base.astype(np.float32), a_min=0.0, a_max=50000.0)
        sequences.append(base)
    return np.asarray(sequences, dtype=np.float32)


def generate_stress_aligned_attack_traffic(n_samples=10000):
    """Generate attack samples from the same family used by stress."""
    if n_samples <= 0:
        return np.zeros((0, 60, 8), dtype=np.float32)
    from src.stress_test.v14.scenarios import TimeSeriesGenerator

    generator = TimeSeriesGenerator()
    category_weights = {
        "ddos": 0.30,
        "portscan": 0.25,
        "exfiltration": 0.20,
        "c2": 0.15,
        "bruteforce": 0.10,
    }
    scenarios = generator.generate(n_samples, category_weights=category_weights, benign_ratio=0.0)
    return np.asarray([scenario.input_data for scenario in scenarios], dtype=np.float32)


def _load_or_generate_timeseries_data(
    base_path: Path,
    normal_cap: int = 50000,
    attack_cap: Optional[int] = None,
    hard_negative_count: int = 5000,
    stress_benign_count: int = 15000,
    stress_hard_negative_count: int = 10000,
    stress_attack_count: int = 12000,
):
    """Load timeseries sources and return sequences, labels, normalization stats, and provenance."""
    source_counts: Dict[str, Dict[str, int]] = {}
    source_details: Dict[str, Any] = {}
    skipped_counts: Dict[str, int] = {}
    source_stats: Dict[str, Dict[str, Any]] = {}
    sample_groups: list[np.ndarray] = []
    label_groups: list[np.ndarray] = []
    source_name_groups: list[np.ndarray] = []

    live_benign_path = base_path / 'datasets' / 'live_benign' / 'timeseries_benign.npy'
    synth_attack_path = base_path / 'datasets' / 'timeseries' / 'attack_traffic_expansion.npy'
    synth_normal_path = base_path / 'datasets' / 'timeseries' / 'normal_traffic_expansion.npy'
    assert_allowed_training_paths(
        [live_benign_path, synth_attack_path, synth_normal_path],
        context="timeseries training data source",
    )

    if live_benign_path.exists():
        print(f"Loading live benign from {live_benign_path}")
        normal = _ensure_timeseries_shape(np.load(live_benign_path)[:normal_cap], "live_benign_timeseries")
        source_details["live_benign_present"] = True
        source_details["normal_source"] = str(live_benign_path)
        _append_source(sample_groups, label_groups, source_name_groups, source_stats, source_counts, "live_benign_timeseries", normal, 0)
    elif synth_normal_path.exists():
        print(f"Loading synthetic normal from {synth_normal_path}")
        normal = _ensure_timeseries_shape(np.load(synth_normal_path, mmap_mode='r')[:normal_cap], "synthetic_normal_expansion")
        source_details["live_benign_present"] = False
        source_details["normal_source"] = str(synth_normal_path)
        _append_source(sample_groups, label_groups, source_name_groups, source_stats, source_counts, "synthetic_normal_expansion", normal, 0)
    else:
        print("Generating normal traffic...")
        normal = _ensure_timeseries_shape(generate_normal_traffic(15000), "generated_normal_traffic")
        source_details["live_benign_present"] = False
        source_details["normal_source"] = "generated_normal_traffic"
        _append_source(sample_groups, label_groups, source_name_groups, source_stats, source_counts, "generated_normal_traffic", normal, 0)

    if stress_benign_count > 0:
        stress_benign = generate_stress_aligned_benign_sequences(stress_benign_count)
        _append_source(
            sample_groups,
            label_groups,
            source_name_groups,
            source_stats,
            source_counts,
            "stress_aligned_benign",
            stress_benign,
            0,
        )

    if hard_negative_count > 0:
        benign_hard = generate_benign_hard_negative_traffic(hard_negative_count)
        _append_source(
            sample_groups,
            label_groups,
            source_name_groups,
            source_stats,
            source_counts,
            "generated_benign_hard_negatives",
            benign_hard,
            0,
        )

    if stress_hard_negative_count > 0:
        stress_hard = generate_stress_aligned_hard_negative_traffic(stress_hard_negative_count)
        _append_source(
            sample_groups,
            label_groups,
            source_name_groups,
            source_stats,
            source_counts,
            "stress_aligned_hard_negatives",
            stress_hard,
            0,
        )

    benign_total = sum(bucket["benign"] for bucket in source_counts.values())
    attack_limit = attack_cap or benign_total
    if synth_attack_path.exists():
        print(f"Loading synthetic attack from {synth_attack_path}")
        attack = _ensure_timeseries_shape(np.load(synth_attack_path, mmap_mode='r')[:attack_limit], "synthetic_attack_expansion")
        source_details["attack_source"] = str(synth_attack_path)
        _append_source(
            sample_groups,
            label_groups,
            source_name_groups,
            source_stats,
            source_counts,
            "synthetic_attack_expansion",
            attack,
            1,
        )
    else:
        print("Generating attack traffic...")
        attack = _ensure_timeseries_shape(generate_attack_traffic(attack_limit), "generated_attack_traffic")
        source_details["attack_source"] = "generated_attack_traffic"
        _append_source(
            sample_groups,
            label_groups,
            source_name_groups,
            source_stats,
            source_counts,
            "generated_attack_traffic",
            attack,
            1,
        )

    if stress_attack_count > 0:
        stress_attack = _ensure_timeseries_shape(generate_stress_aligned_attack_traffic(stress_attack_count), "stress_aligned_attack")
        _append_source(
            sample_groups,
            label_groups,
            source_name_groups,
            source_stats,
            source_counts,
            "stress_aligned_attack",
            stress_attack,
            1,
        )

    all_data = np.concatenate(sample_groups, axis=0)
    all_data, mins, maxs = normalize_data(all_data)
    labels = np.concatenate(label_groups, axis=0)
    source_names = np.concatenate(source_name_groups, axis=0)

    idx = np.random.permutation(len(all_data))
    sequences = all_data[idx]
    labels = labels[idx]
    source_names = source_names[idx]

    manifest_sources = {
        "source_counts": source_counts,
        "source_details": source_details,
        "source_stats": source_stats,
        "skipped_counts": skipped_counts,
        "totals": {
            "total": int(len(sequences)),
            "malicious": int(labels.sum()),
            "benign": int(len(labels) - labels.sum()),
        },
    }
    return sequences, labels, mins, maxs, source_names, manifest_sources

	
def generate_normal_traffic(n_samples=10000, seq_len=60, n_features=8):
    """Generate synthetic normal network traffic patterns with realistic noise."""
    sequences = []
    
    for _ in range(n_samples):
        # Base pattern with daily/hourly cycles
        t = np.linspace(0, 4*np.pi, seq_len)
        
        # Add random phase shift for variety
        phase = np.random.uniform(0, 2*np.pi)
        
        # Feature 0: Packet count (cyclic with noise)
        packets = 100 + 30*np.sin(t + phase) + np.random.normal(0, 15, seq_len)
        
        # Feature 1: Bytes transferred
        bytes_tx = packets * np.random.uniform(500, 1500) + np.random.normal(0, 2000, seq_len)
        
        # Feature 2: Unique source IPs
        unique_ips = 20 + 5*np.sin(t/2 + phase) + np.random.normal(0, 5, seq_len)
        
        # Feature 3: Unique destination ports
        unique_ports = 10 + 3*np.sin(t/3 + phase) + np.random.normal(0, 3, seq_len)
        
        # Feature 4: Average packet size
        avg_pkt_size = bytes_tx / (packets + 1)
        
        # Feature 5: TCP ratio (with occasional variation)
        tcp_ratio = 0.7 + np.random.normal(0, 0.15, seq_len)
        
        # Feature 6: SYN flag ratio (can spike occasionally in normal traffic)
        syn_ratio = 0.1 + np.random.normal(0, 0.05, seq_len)
        # Occasional normal spikes (e.g., new connections burst)
        if np.random.random() < 0.2:
            spike_idx = np.random.randint(10, 50)
            syn_ratio[spike_idx:spike_idx+5] += np.random.uniform(0.1, 0.2)
        
        # Feature 7: Error rate
        error_rate = 0.01 + np.random.exponential(0.01, seq_len)
        
        seq = np.stack([packets, bytes_tx, unique_ips, unique_ports, 
                       avg_pkt_size, tcp_ratio, syn_ratio, error_rate], axis=1)
        
        # Add global noise to make patterns less distinct
        seq += np.random.normal(0, 0.05 * np.abs(seq.mean()), seq.shape)
        
        sequences.append(seq)
    
    return np.array(sequences, dtype=np.float32)


def generate_attack_traffic(n_samples=10000, seq_len=60, n_features=8):
    """Generate synthetic attack traffic patterns - more subtle and realistic."""
    sequences = []
    attack_types = ['ddos', 'portscan', 'bruteforce', 'exfiltration', 'subtle_ddos', 'slow_scan']
    
    for _ in range(n_samples):
        attack = np.random.choice(attack_types)
        t = np.linspace(0, 4*np.pi, seq_len)
        phase = np.random.uniform(0, 2*np.pi)
        
        # Start with normal-looking baseline
        packets = 100 + 30*np.sin(t + phase) + np.random.normal(0, 15, seq_len)
        unique_ips = 20 + 5*np.sin(t/2 + phase) + np.random.normal(0, 5, seq_len)
        unique_ports = 10 + 3*np.sin(t/3 + phase) + np.random.normal(0, 3, seq_len)
        syn_ratio = 0.1 + np.random.normal(0, 0.05, seq_len)
        error_rate = 0.01 + np.random.exponential(0.01, seq_len)
        tcp_ratio = 0.7 + np.random.normal(0, 0.15, seq_len)
        
        if attack == 'ddos':
            # Sudden spike in packets, many source IPs
            spike_start = np.random.randint(20, 40)
            packets[spike_start:] += np.random.uniform(200, 800)
            unique_ips[spike_start:] += np.random.uniform(50, 200)
            syn_ratio[spike_start:] = np.random.uniform(0.4, 0.7)
            
        elif attack == 'subtle_ddos':
            # Gradual increase - harder to detect
            spike_start = np.random.randint(15, 30)
            ramp = np.linspace(0, 1, seq_len - spike_start)
            packets[spike_start:] += ramp * np.random.uniform(100, 300)
            unique_ips[spike_start:] += ramp * np.random.uniform(30, 80)
            syn_ratio[spike_start:] += ramp * np.random.uniform(0.1, 0.3)
            
        elif attack == 'portscan':
            # Many destination ports, moderate packet count
            scan_start = np.random.randint(10, 30)
            unique_ports[scan_start:] += np.random.uniform(30, 100)
            syn_ratio[scan_start:] = np.random.uniform(0.5, 0.8)
            
        elif attack == 'slow_scan':
            # Very slow port scan - subtle increase over time
            unique_ports += np.linspace(0, np.random.uniform(20, 50), seq_len)
            syn_ratio += np.linspace(0, np.random.uniform(0.1, 0.2), seq_len)
            
        elif attack == 'bruteforce':
            # Repeated connections, high error rate
            bf_start = np.random.randint(15, 35)
            error_rate[bf_start:] = np.random.uniform(0.15, 0.4)
            unique_ports[bf_start:] = 1 + np.random.normal(0, 0.5, seq_len - bf_start)
            
        else:  # exfiltration
            # Gradual increase in outbound bytes
            exfil_start = np.random.randint(20, 40)
            packets[exfil_start:] *= np.linspace(1, np.random.uniform(2, 5), seq_len - exfil_start)
        
        bytes_tx = packets * np.random.uniform(500, 1500) + np.random.normal(0, 2000, seq_len)
        avg_pkt_size = bytes_tx / (packets + 1)
        
        seq = np.stack([packets, bytes_tx, unique_ips, unique_ports,
                       avg_pkt_size, tcp_ratio, syn_ratio, error_rate], axis=1)
        
        # Add noise to make patterns less obvious
        seq += np.random.normal(0, 0.05 * np.abs(seq.mean()), seq.shape)
        
        sequences.append(seq)
    
    return np.array(sequences, dtype=np.float32)


def normalize_data(data):
    """Normalize features to 0-1 range."""
    mins = data.min(axis=(0, 1), keepdims=True)
    maxs = data.max(axis=(0, 1), keepdims=True)
    normalized = (data - mins) / (maxs - mins + 1e-8)
    return normalized, mins, maxs


def train():
    """Main training function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint-every', type=int, default=500, help='Save checkpoint every N batches')
    parser.add_argument('--epochs', type=int, default=60, help='Training epochs (default: 60)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4, help='Dataloader workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible smoke runs')
    parser.add_argument('--base-path', type=str, default=None, help='Optional repo root override')
    parser.add_argument('--normal-cap', type=int, default=50000, help='Cap for benign timeseries samples')
    parser.add_argument('--attack-cap', type=int, default=None, help='Optional cap for attack timeseries samples')
    parser.add_argument('--hard-negative-count', type=int, default=5000, help='Generated benign hard negatives')
    parser.add_argument('--stress-benign-count', type=int, default=15000, help='Stress-aligned benign sequences')
    parser.add_argument('--stress-hard-negative-count', type=int, default=10000, help='Stress-aligned benign hard negatives')
    parser.add_argument('--stress-attack-count', type=int, default=12000, help='Stress-aligned attack supplement')
    args = parser.parse_args()
    
    base_path = Path(args.base_path) if args.base_path else Path(__file__).parent.parent.parent
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = setup_gpu()
    
    # Checkpoint manager
    ckpt_dir = base_path / 'checkpoints' / 'timeseries'
    ckpt_mgr = CheckpointManager(str(ckpt_dir), 'timeseries_lstm', args.checkpoint_every)
    
    # Load or generate data
    print("\n--- Loading/Generating Data ---")
    sequences, labels, mins, maxs, source_names, data_summary = _load_or_generate_timeseries_data(
        base_path,
        normal_cap=args.normal_cap,
        attack_cap=args.attack_cap,
        hard_negative_count=args.hard_negative_count,
        stress_benign_count=args.stress_benign_count,
        stress_hard_negative_count=args.stress_hard_negative_count,
        stress_attack_count=args.stress_attack_count,
    )
    
    print(f"Total: {len(sequences)} sequences ({sum(labels==0):.0f} normal, {sum(labels==1):.0f} attack)")
    
    # Create dataset
    dataset = TimeSeriesDataset(sequences, labels)
    split_labels = np.array([f"{int(label)}::{source}" for label, source in zip(labels.astype(int), source_names)], dtype=object)
    train_idx, val_idx = stratified_index_split(split_labels, test_size=0.2, seed=args.seed)
    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    
    persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        timeout=0,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        timeout=0,
        persistent_workers=persistent_workers,
    )
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    print("\n--- Creating Model ---")
    model = TimeSeriesLSTM(input_dim=8).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Training setup
    pos_count = max(float(labels.sum()), 1.0)
    neg_count = max(float(len(labels) - labels.sum()), 1.0)
    pos_weight = torch.tensor([neg_count / pos_count], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler = GradScaler()
    early_stop = EarlyStopping(patience=7)
    eval_threshold = load_operational_threshold("timeseries", default=0.5)
    
    # Resume from checkpoint only when explicitly requested.
    start_epoch, start_batch, global_step = 0, 0, 0
    if args.resume:
        resume_info = ckpt_mgr.load(model, optimizer, scheduler, scaler, device)
        start_epoch = resume_info['epoch']
        start_batch = resume_info['batch_idx']
        global_step = resume_info['global_step']
        if start_batch >= len(train_loader):
            start_epoch += 1
            start_batch = 0
        if start_epoch >= args.epochs:
            print(
                f"Resume point epoch {start_epoch} is at or beyond requested --epochs {args.epochs}; "
                "saving current state without additional training."
            )
    
    # Training loop
    print("\n--- Training ---")
    best_metrics = None
    best_state = model.state_dict().copy()
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        train_loss = 0
        batches_processed = 0
        
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    initial=epoch_start_batch, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in pbar:
            if batch_idx < epoch_start_batch:
                continue
                
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            batches_processed += 1
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if ckpt_mgr.should_save(batch_idx):
                ckpt_mgr.save(epoch, batch_idx, model, optimizer, scheduler, scaler, global_step)
        
        ckpt_mgr.save(epoch, len(train_loader), model, optimizer, scheduler, scaler, global_step)
        
        train_loss /= max(batches_processed, 1)
        
        # Validate
        model.eval()
        val_loss = 0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_targets.extend(targets.detach().cpu().numpy().tolist())
        
        val_loss /= len(val_loader)
        val_metrics = binary_metrics(all_probs, all_targets, eval_threshold)
        if val_metrics["tp"] == 0:
            raise RuntimeError("Timeseries validation collapsed to zero true positives; aborting training.")
        
        scheduler.step(val_loss)
        
        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}, val_recall={val_metrics['recall']:.4f}, val_fpr={val_metrics['fpr']:.4f}"
        )
        
        current_score = (
            val_metrics['f1'],
            val_metrics['recall'],
            -val_metrics['fpr'],
        )
        best_score = (
            best_metrics['f1'],
            best_metrics['recall'],
            -best_metrics['fpr'],
        ) if best_metrics is not None else None
        if best_metrics is None or current_score > best_score:
            best_metrics = dict(val_metrics)
            best_state = model.state_dict().copy()
        
        if early_stop(val_loss):
            print("Early stopping triggered")
            break
    
    # Save best model
    print("\n--- Saving Model ---")
    model.load_state_dict(best_state)
    models_dir = base_path / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model.eval()
    example = torch.zeros(1, 60, 8, dtype=torch.float32).to(device)
    save_model(model, models_dir / 'timeseries_lstm', example)
    torch.save(best_state, models_dir / 'timeseries_lstm.pth')
    # Save normalization stats for inference parity
    np.savez(models_dir / 'timeseries_norm_v1.npz', mins=mins, maxs=maxs)
    write_training_manifest(
        models_dir / 'timeseries_lstm_training_manifest.json',
        {
            "model": "timeseries",
            "dataset_size": len(dataset),
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "operational_threshold": eval_threshold,
            "best_metrics": best_metrics or {},
            "label_counts": {
                "malicious": int(labels.sum()),
                "benign": int(len(labels) - labels.sum()),
            },
            "pos_weight": float(pos_weight.item()),
            "source_counts": data_summary["source_counts"],
            "source_details": data_summary["source_details"],
            "source_stats": data_summary["source_stats"],
            "skipped_counts": data_summary["skipped_counts"],
            "python_executable": sys.executable,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "stress_benign_count": args.stress_benign_count,
            "stress_hard_negative_count": args.stress_hard_negative_count,
            "stress_attack_count": args.stress_attack_count,
            "validation_sources": {
                "train": {source: int(np.sum(source_names[train_idx] == source)) for source in np.unique(source_names)},
                "val": {source: int(np.sum(source_names[val_idx] == source)) for source in np.unique(source_names)},
            },
            "normalization_shape": {
                "mins": list(np.asarray(mins).shape),
                "maxs": list(np.asarray(maxs).shape),
            },
        },
    )
    
    print(f"✓ Model saved to models/timeseries_lstm.pt")
    print(f"✓ Best validation F1: {(best_metrics or {}).get('f1', 0.0):.4f}")


if __name__ == "__main__":
    train()
