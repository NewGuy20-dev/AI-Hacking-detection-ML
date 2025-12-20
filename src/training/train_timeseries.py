"""Train Time-Series LSTM for network traffic anomaly detection."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
from tqdm import tqdm

from torch_models.timeseries_lstm import TimeSeriesLSTM
from torch_models.datasets import TimeSeriesDataset
from torch_models.utils import setup_gpu, EarlyStopping, save_model


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
    return (data - mins) / (maxs - mins + 1e-8)


def train():
    """Main training function."""
    device = setup_gpu()
    base_path = Path(__file__).parent.parent.parent
    
    # Generate data
    print("\n--- Generating Data ---")
    normal = generate_normal_traffic(15000)
    attack = generate_attack_traffic(15000)
    
    # Normalize
    all_data = np.concatenate([normal, attack], axis=0)
    all_data = normalize_data(all_data)
    normal = all_data[:15000]
    attack = all_data[15000:]
    
    # Create labels
    sequences = np.concatenate([normal, attack], axis=0)
    labels = np.array([0]*len(normal) + [1]*len(attack), dtype=np.float32)
    
    # Shuffle
    idx = np.random.permutation(len(sequences))
    sequences, labels = sequences[idx], labels[idx]
    
    print(f"Total: {len(sequences)} sequences ({sum(labels==0):.0f} normal, {sum(labels==1):.0f} attack)")
    
    # Create dataset
    dataset = TimeSeriesDataset(sequences, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Model
    print("\n--- Creating Model ---")
    model = TimeSeriesLSTM(input_dim=8).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")
    
    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    scaler = GradScaler()
    early_stop = EarlyStopping(patience=7)
    
    # Training loop
    print("\n--- Training ---")
    best_val_acc = 0
    best_state = None
    
    for epoch in range(60):  # Increased from 50
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
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
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
    
    print(f"✓ Model saved to models/timeseries_lstm.pt")
    print(f"✓ Best validation accuracy: {best_val_acc:.2%}")


if __name__ == "__main__":
    train()
