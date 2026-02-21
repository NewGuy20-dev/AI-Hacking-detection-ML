"""Phase A4: Improved time-series synthetic data with subtle attack patterns."""
import numpy as np
from pathlib import Path


def generate_normal_traffic(n_samples=15000, seq_len=60, n_features=8):
    """Generate realistic normal network traffic with natural variations."""
    sequences = []
    
    for _ in range(n_samples):
        t = np.linspace(0, 4*np.pi, seq_len)
        
        # Add realistic variations: business hours, weekends, maintenance
        variation = np.random.choice(['normal', 'busy', 'quiet', 'maintenance'])
        base_mult = {'normal': 1.0, 'busy': 1.5, 'quiet': 0.6, 'maintenance': 0.3}[variation]
        
        packets = (100 + 30*np.sin(t) + np.random.normal(0, 15, seq_len)) * base_mult
        bytes_tx = packets * np.random.uniform(500, 1500) + np.random.normal(0, 1000, seq_len)
        unique_ips = (20 + 5*np.sin(t/2) + np.random.normal(0, 4, seq_len)) * base_mult
        unique_ports = 10 + 3*np.sin(t/3) + np.random.normal(0, 3, seq_len)
        avg_pkt_size = bytes_tx / (packets + 1)
        tcp_ratio = np.clip(0.7 + np.random.normal(0, 0.1, seq_len), 0.4, 0.95)
        syn_ratio = np.clip(0.1 + np.random.normal(0, 0.03, seq_len), 0.02, 0.25)
        error_rate = np.clip(0.01 + np.random.exponential(0.008, seq_len), 0, 0.1)
        
        # Occasional legitimate spikes (backup, updates)
        if np.random.random() < 0.1:
            spike_start = np.random.randint(10, 40)
            spike_len = np.random.randint(5, 15)
            packets[spike_start:spike_start+spike_len] *= 1.8
            bytes_tx[spike_start:spike_start+spike_len] *= 2.0
        
        seq = np.stack([packets, bytes_tx, unique_ips, unique_ports,
                       avg_pkt_size, tcp_ratio, syn_ratio, error_rate], axis=1)
        sequences.append(seq)
    
    return np.array(sequences, dtype=np.float32)


def generate_subtle_attack_traffic(n_samples=15000, seq_len=60, n_features=8):
    """Generate subtle attack patterns that are harder to detect."""
    sequences = []
    attack_types = ['slow_ddos', 'stealthy_scan', 'low_bruteforce', 'slow_exfil', 'intermittent']
    
    for _ in range(n_samples):
        attack = np.random.choice(attack_types)
        t = np.linspace(0, 4*np.pi, seq_len)
        
        # Start with normal baseline
        packets = 100 + 30*np.sin(t) + np.random.normal(0, 15, seq_len)
        bytes_tx = packets * np.random.uniform(500, 1500) + np.random.normal(0, 1000, seq_len)
        unique_ips = 20 + 5*np.sin(t/2) + np.random.normal(0, 4, seq_len)
        unique_ports = 10 + 3*np.sin(t/3) + np.random.normal(0, 3, seq_len)
        tcp_ratio = np.clip(0.7 + np.random.normal(0, 0.1, seq_len), 0.4, 0.95)
        syn_ratio = np.clip(0.1 + np.random.normal(0, 0.03, seq_len), 0.02, 0.25)
        error_rate = np.clip(0.01 + np.random.exponential(0.008, seq_len), 0, 0.1)
        
        attack_start = np.random.randint(15, 35)
        
        if attack == 'slow_ddos':
            # Gradual increase (50-100% over time, not 10x spike)
            ramp = np.linspace(1, np.random.uniform(1.5, 2.0), seq_len - attack_start)
            packets[attack_start:] *= ramp
            unique_ips[attack_start:] += np.linspace(0, np.random.uniform(20, 50), seq_len - attack_start)
            syn_ratio[attack_start:] = np.clip(syn_ratio[attack_start:] + 0.15, 0, 0.5)
            
        elif attack == 'stealthy_scan':
            # Low and slow port scan (10-30 new ports, not 200)
            unique_ports[attack_start:] += np.random.uniform(10, 30)
            syn_ratio[attack_start:] = np.clip(syn_ratio[attack_start:] + 0.1, 0, 0.4)
            packets[attack_start:] *= 0.8  # Slightly reduced traffic
            
        elif attack == 'low_bruteforce':
            # Subtle error rate increase (5-15%, not 30-70%)
            error_rate[attack_start:] = np.clip(
                error_rate[attack_start:] + np.random.uniform(0.05, 0.15), 0, 0.25)
            unique_ports[attack_start:] = np.clip(unique_ports[attack_start:] - 5, 1, 50)
            
        elif attack == 'slow_exfil':
            # Gradual outbound increase (2-3x, not 10x)
            ramp = np.linspace(1, np.random.uniform(2, 3), seq_len - attack_start)
            bytes_tx[attack_start:] *= ramp
            unique_ips[attack_start:] = np.clip(unique_ips[attack_start:] - 10, 2, 50)
            
        else:  # intermittent
            # Attack, pause, attack pattern
            for i in range(attack_start, seq_len, 10):
                end = min(i + 5, seq_len)
                packets[i:end] *= 1.5
                syn_ratio[i:end] += 0.1
        
        avg_pkt_size = bytes_tx / (packets + 1)
        
        seq = np.stack([packets, bytes_tx, unique_ips, unique_ports,
                       avg_pkt_size, tcp_ratio, syn_ratio, error_rate], axis=1)
        sequences.append(seq)
    
    return np.array(sequences, dtype=np.float32)


def main():
    base_path = Path(__file__).parent.parent
    data_dir = base_path / 'datasets' / 'timeseries'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating improved time-series data...")
    
    normal = generate_normal_traffic(15000)
    attack = generate_subtle_attack_traffic(15000)
    
    np.save(data_dir / 'normal_traffic_improved.npy', normal)
    np.save(data_dir / 'attack_traffic_improved.npy', attack)
    
    print(f"  ✓ Saved {len(normal)} normal sequences")
    print(f"  ✓ Saved {len(attack)} subtle attack sequences")
    print("\nDone! These patterns are more realistic and harder to detect.")


if __name__ == "__main__":
    main()
