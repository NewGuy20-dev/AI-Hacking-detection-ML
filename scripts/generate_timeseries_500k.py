#!/usr/bin/env python3
"""Generate 500k synthetic timeseries sequences for LSTM training."""
import numpy as np
from pathlib import Path
from tqdm import tqdm

SEQ_LEN = 60
N_FEATURES = 8

def generate_normal_batch(n_samples):
    """Generate normal traffic patterns."""
    sequences = np.zeros((n_samples, SEQ_LEN, N_FEATURES), dtype=np.float32)
    
    for i in range(n_samples):
        t = np.linspace(0, 4*np.pi, SEQ_LEN)
        
        # Feature 0: Packet rate (smooth with daily pattern)
        base = 50 + 20 * np.sin(t) + np.random.normal(0, 3, SEQ_LEN)
        sequences[i, :, 0] = np.clip(base, 10, 100)
        
        # Feature 1: Byte rate (correlated with packet rate)
        sequences[i, :, 1] = sequences[i, :, 0] * np.random.uniform(800, 1200) + np.random.normal(0, 500, SEQ_LEN)
        
        # Feature 2: Connection count (stable)
        sequences[i, :, 2] = np.random.uniform(20, 80) + np.random.normal(0, 5, SEQ_LEN)
        
        # Feature 3: Error rate (low)
        sequences[i, :, 3] = np.clip(np.random.exponential(0.02, SEQ_LEN), 0, 0.2)
        
        # Feature 4: Unique IPs (moderate variation)
        sequences[i, :, 4] = np.random.uniform(10, 50) + np.random.normal(0, 3, SEQ_LEN)
        
        # Feature 5: Avg packet size (stable)
        sequences[i, :, 5] = np.random.uniform(400, 1200) + np.random.normal(0, 50, SEQ_LEN)
        
        # Feature 6: Protocol distribution (stable)
        sequences[i, :, 6] = np.random.uniform(0.6, 0.9) + np.random.normal(0, 0.05, SEQ_LEN)
        
        # Feature 7: Port entropy (moderate)
        sequences[i, :, 7] = np.random.uniform(2, 4) + np.random.normal(0, 0.2, SEQ_LEN)
    
    return sequences

def generate_ddos_batch(n_samples):
    """Generate DDoS attack patterns."""
    sequences = np.zeros((n_samples, SEQ_LEN, N_FEATURES), dtype=np.float32)
    
    for i in range(n_samples):
        attack_start = np.random.randint(10, 40)
        
        # Feature 0: Packet rate (spike during attack)
        base = np.ones(SEQ_LEN) * 50
        base[attack_start:] = np.random.uniform(500, 2000)
        sequences[i, :, 0] = base + np.random.normal(0, 20, SEQ_LEN)
        
        # Feature 1: Byte rate (massive spike)
        sequences[i, :, 1] = sequences[i, :, 0] * np.random.uniform(500, 800)
        
        # Feature 2: Connection count (explosion)
        conn = np.ones(SEQ_LEN) * 50
        conn[attack_start:] = np.random.uniform(1000, 5000)
        sequences[i, :, 2] = conn
        
        # Feature 3: Error rate (high during attack)
        err = np.ones(SEQ_LEN) * 0.02
        err[attack_start:] = np.random.uniform(0.3, 0.8)
        sequences[i, :, 3] = err
        
        # Feature 4: Unique IPs (massive increase)
        ips = np.ones(SEQ_LEN) * 30
        ips[attack_start:] = np.random.uniform(500, 2000)
        sequences[i, :, 4] = ips
        
        # Feature 5: Avg packet size (small packets in DDoS)
        size = np.ones(SEQ_LEN) * 800
        size[attack_start:] = np.random.uniform(40, 200)
        sequences[i, :, 5] = size
        
        # Feature 6: Protocol distribution (skewed)
        proto = np.ones(SEQ_LEN) * 0.75
        proto[attack_start:] = np.random.uniform(0.1, 0.3)
        sequences[i, :, 6] = proto
        
        # Feature 7: Port entropy (low - targeting specific ports)
        ent = np.ones(SEQ_LEN) * 3
        ent[attack_start:] = np.random.uniform(0.5, 1.5)
        sequences[i, :, 7] = ent
    
    return sequences

def generate_portscan_batch(n_samples):
    """Generate port scanning patterns."""
    sequences = np.zeros((n_samples, SEQ_LEN, N_FEATURES), dtype=np.float32)
    
    for i in range(n_samples):
        scan_start = np.random.randint(5, 30)
        
        # Feature 0: Packet rate (moderate increase)
        base = np.ones(SEQ_LEN) * 50
        base[scan_start:] = np.random.uniform(100, 300)
        sequences[i, :, 0] = base + np.random.normal(0, 10, SEQ_LEN)
        
        # Feature 1: Byte rate (low - small probe packets)
        sequences[i, :, 1] = sequences[i, :, 0] * np.random.uniform(40, 100)
        
        # Feature 2: Connection count (many short connections)
        conn = np.ones(SEQ_LEN) * 50
        conn[scan_start:] = np.random.uniform(200, 500)
        sequences[i, :, 2] = conn
        
        # Feature 3: Error rate (high - many refused)
        err = np.ones(SEQ_LEN) * 0.02
        err[scan_start:] = np.random.uniform(0.5, 0.9)
        sequences[i, :, 3] = err
        
        # Feature 4: Unique IPs (low - single scanner)
        sequences[i, :, 4] = np.random.uniform(1, 5) + np.random.normal(0, 0.5, SEQ_LEN)
        
        # Feature 5: Avg packet size (very small)
        size = np.ones(SEQ_LEN) * 800
        size[scan_start:] = np.random.uniform(40, 80)
        sequences[i, :, 5] = size
        
        # Feature 6: Protocol distribution (TCP heavy)
        proto = np.ones(SEQ_LEN) * 0.75
        proto[scan_start:] = np.random.uniform(0.95, 1.0)
        sequences[i, :, 6] = proto
        
        # Feature 7: Port entropy (very high - scanning many ports)
        ent = np.ones(SEQ_LEN) * 3
        ent[scan_start:] = np.random.uniform(6, 10)
        sequences[i, :, 7] = ent
    
    return sequences

def generate_exfiltration_batch(n_samples):
    """Generate data exfiltration patterns."""
    sequences = np.zeros((n_samples, SEQ_LEN, N_FEATURES), dtype=np.float32)
    
    for i in range(n_samples):
        exfil_start = np.random.randint(10, 40)
        
        # Feature 0: Packet rate (moderate)
        sequences[i, :, 0] = 50 + np.random.normal(0, 5, SEQ_LEN)
        
        # Feature 1: Byte rate (high outbound - data leaving)
        base = np.ones(SEQ_LEN) * 50000
        base[exfil_start:] = np.random.uniform(500000, 2000000)
        sequences[i, :, 1] = base + np.random.normal(0, 10000, SEQ_LEN)
        
        # Feature 2: Connection count (few persistent connections)
        sequences[i, :, 2] = np.random.uniform(5, 20) + np.random.normal(0, 2, SEQ_LEN)
        
        # Feature 3: Error rate (low - successful transfers)
        sequences[i, :, 3] = np.clip(np.random.exponential(0.01, SEQ_LEN), 0, 0.1)
        
        # Feature 4: Unique IPs (very few - C2 servers)
        sequences[i, :, 4] = np.random.uniform(1, 3) + np.random.normal(0, 0.3, SEQ_LEN)
        
        # Feature 5: Avg packet size (large - data chunks)
        size = np.ones(SEQ_LEN) * 800
        size[exfil_start:] = np.random.uniform(1400, 1500)  # MTU
        sequences[i, :, 5] = size
        
        # Feature 6: Protocol distribution (encrypted traffic)
        sequences[i, :, 6] = np.random.uniform(0.8, 0.95) + np.random.normal(0, 0.02, SEQ_LEN)
        
        # Feature 7: Port entropy (low - single destination)
        ent = np.ones(SEQ_LEN) * 3
        ent[exfil_start:] = np.random.uniform(0.5, 1.5)
        sequences[i, :, 7] = ent
    
    return sequences

def generate_bruteforce_batch(n_samples):
    """Generate brute force attack patterns."""
    sequences = np.zeros((n_samples, SEQ_LEN, N_FEATURES), dtype=np.float32)
    
    for i in range(n_samples):
        attack_start = np.random.randint(5, 25)
        
        # Feature 0: Packet rate (steady high)
        base = np.ones(SEQ_LEN) * 50
        base[attack_start:] = np.random.uniform(150, 400)
        sequences[i, :, 0] = base + np.random.normal(0, 10, SEQ_LEN)
        
        # Feature 1: Byte rate (moderate - login attempts)
        sequences[i, :, 1] = sequences[i, :, 0] * np.random.uniform(200, 500)
        
        # Feature 2: Connection count (many attempts)
        conn = np.ones(SEQ_LEN) * 50
        conn[attack_start:] = np.random.uniform(100, 300)
        sequences[i, :, 2] = conn
        
        # Feature 3: Error rate (high - failed logins)
        err = np.ones(SEQ_LEN) * 0.02
        err[attack_start:] = np.random.uniform(0.7, 0.99)
        sequences[i, :, 3] = err
        
        # Feature 4: Unique IPs (few - attacker IPs)
        sequences[i, :, 4] = np.random.uniform(1, 10) + np.random.normal(0, 1, SEQ_LEN)
        
        # Feature 5: Avg packet size (small - credentials)
        size = np.ones(SEQ_LEN) * 800
        size[attack_start:] = np.random.uniform(100, 300)
        sequences[i, :, 5] = size
        
        # Feature 6: Protocol distribution (TCP)
        sequences[i, :, 6] = np.random.uniform(0.9, 1.0) + np.random.normal(0, 0.02, SEQ_LEN)
        
        # Feature 7: Port entropy (very low - single service)
        ent = np.ones(SEQ_LEN) * 3
        ent[attack_start:] = np.random.uniform(0.1, 0.5)
        sequences[i, :, 7] = ent
    
    return sequences

def main():
    output_dir = Path(__file__).parent.parent / "datasets" / "timeseries"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = 500_000
    normal_count = 250_000
    attack_count = 250_000  # Split among attack types
    
    print(f"Generating {total:,} timeseries sequences...")
    print(f"Sequence length: {SEQ_LEN}, Features: {N_FEATURES}")
    
    batch_size = 10000
    
    # Generate normal sequences
    print("\nGenerating normal traffic...")
    normal_sequences = []
    for i in tqdm(range(0, normal_count, batch_size)):
        batch = min(batch_size, normal_count - i)
        normal_sequences.append(generate_normal_batch(batch))
    normal_data = np.concatenate(normal_sequences, axis=0)
    
    # Generate attack sequences
    attack_generators = [
        ("DDoS", generate_ddos_batch, 80_000),
        ("Portscan", generate_portscan_batch, 60_000),
        ("Exfiltration", generate_exfiltration_batch, 60_000),
        ("Bruteforce", generate_bruteforce_batch, 50_000),
    ]
    
    attack_sequences = []
    for name, gen_func, count in attack_generators:
        print(f"\nGenerating {name}...")
        for i in tqdm(range(0, count, batch_size)):
            batch = min(batch_size, count - i)
            attack_sequences.append(gen_func(batch))
    
    attack_data = np.concatenate(attack_sequences, axis=0)
    
    # Save
    normal_path = output_dir / "normal_traffic_500k.npy"
    attack_path = output_dir / "attack_traffic_500k.npy"
    
    np.save(normal_path, normal_data)
    np.save(attack_path, attack_data)
    
    print(f"\n✓ Generated {normal_count:,} normal sequences ({normal_data.nbytes/1024/1024:.1f} MB)")
    print(f"✓ Generated {attack_count:,} attack sequences ({attack_data.nbytes/1024/1024:.1f} MB)")
    print(f"  DDoS: 80k, Portscan: 60k, Exfiltration: 60k, Bruteforce: 50k")
    print(f"\nSaved to:")
    print(f"  {normal_path}")
    print(f"  {attack_path}")

if __name__ == "__main__":
    main()
