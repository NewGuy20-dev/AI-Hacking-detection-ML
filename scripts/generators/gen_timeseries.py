#!/usr/bin/env python3
"""Generate 5M timeseries samples (shape: samples, 60, 8)."""
import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from generators.utils import ProgressTracker, RatioSampler

def gen_normal_traffic(n_samples: int) -> np.ndarray:
    """Generate normal network traffic patterns."""
    data = np.zeros((n_samples, 60, 8), dtype=np.float32)
    for i in range(n_samples):
        # Feature 0: packet count (low, steady)
        data[i, :, 0] = np.random.uniform(10, 100, 60) + np.sin(np.linspace(0, 4*np.pi, 60)) * 20
        # Feature 1: byte count
        data[i, :, 1] = data[i, :, 0] * np.random.uniform(500, 1500)
        # Feature 2: unique IPs (low)
        data[i, :, 2] = np.random.uniform(5, 30, 60)
        # Feature 3: unique ports (low)
        data[i, :, 3] = np.random.uniform(3, 20, 60)
        # Feature 4: avg packet size
        data[i, :, 4] = np.random.uniform(500, 1500, 60)
        # Feature 5: connection duration
        data[i, :, 5] = np.random.uniform(1, 60, 60)
        # Feature 6: error rate (low)
        data[i, :, 6] = np.random.uniform(0, 0.05, 60)
        # Feature 7: retransmission rate (low)
        data[i, :, 7] = np.random.uniform(0, 0.03, 60)
    return data

def gen_attack_traffic(n_samples: int) -> np.ndarray:
    """Generate attack traffic patterns."""
    data = np.zeros((n_samples, 60, 8), dtype=np.float32)
    for i in range(n_samples):
        attack = np.random.choice(['ddos', 'portscan', 'exfil', 'bruteforce'])
        
        if attack == 'ddos':
            # High packet count spike
            data[i, :, 0] = np.random.uniform(500, 2000, 60)
            data[i, 20:40, 0] *= np.random.uniform(5, 20)  # Spike
            data[i, :, 1] = data[i, :, 0] * np.random.uniform(100, 500)  # Small packets
            data[i, :, 2] = np.random.uniform(100, 1000, 60)  # Many IPs
            data[i, :, 6] = np.random.uniform(0.3, 0.8, 60)  # High error
        elif attack == 'portscan':
            data[i, :, 0] = np.random.uniform(50, 200, 60)
            data[i, :, 3] = np.random.uniform(100, 500, 60)  # Many ports
            data[i, :, 4] = np.random.uniform(40, 100, 60)  # Small packets
            data[i, :, 5] = np.random.uniform(0.01, 0.5, 60)  # Short connections
        elif attack == 'exfil':
            data[i, :, 0] = np.random.uniform(20, 80, 60)
            data[i, :, 1] = np.random.uniform(50000, 500000, 60)  # Large bytes
            data[i, :, 4] = np.random.uniform(1400, 1500, 60)  # Max packet size
            data[i, :, 5] = np.random.uniform(60, 300, 60)  # Long connections
        else:  # bruteforce
            data[i, :, 0] = np.random.uniform(100, 500, 60)
            data[i, :, 2] = np.random.uniform(1, 5, 60)  # Few IPs
            data[i, :, 3] = np.random.uniform(1, 3, 60)  # Few ports
            data[i, :, 6] = np.random.uniform(0.5, 0.95, 60)  # High error (failed logins)
    return data

def generate_samples(n_benign: int, n_malicious: int, output_dir: Path, batch_size: int = 500000):
    """Generate and save samples."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {n_benign:,} normal traffic samples...")
    normal = gen_normal_traffic(n_benign)
    normal_path = output_dir / "normal_traffic_expansion.npy"
    np.save(normal_path, normal)
    print(f"Saved: {normal_path} ({normal.nbytes / 1e9:.2f} GB)")
    
    print(f"Generating {n_malicious:,} attack traffic samples...")
    attack = gen_attack_traffic(n_malicious)
    attack_path = output_dir / "attack_traffic_expansion.npy"
    np.save(attack_path, attack)
    print(f"Saved: {attack_path} ({attack.nbytes / 1e9:.2f} GB)")

def main():
    parser = argparse.ArgumentParser(description="Generate timeseries samples")
    parser.add_argument("--total", type=int, default=5_000_000, help="Total samples")
    parser.add_argument("--output", type=Path, default=Path("datasets/timeseries"))
    args = parser.parse_args()
    
    sampler = RatioSampler(args.total)
    n_benign, n_malicious = sampler.get_counts()
    print(f"Generating {n_benign:,} normal + {n_malicious:,} attack = {args.total:,} total")
    print(f"Expected size: ~{args.total * 60 * 8 * 4 / 1e9:.1f} GB")
    
    generate_samples(n_benign, n_malicious, args.output)

if __name__ == "__main__":
    main()
