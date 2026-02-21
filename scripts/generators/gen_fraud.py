#!/usr/bin/env python3
"""Generate 5M fraud samples (30 features matching scaler)."""
import argparse
import random
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from generators.utils import ProgressTracker, RatioSampler, append_to_jsonl

def gen_benign_transaction():
    """Generate benign transaction (30 features like creditcard.csv)."""
    # Time feature (seconds from start)
    time = random.uniform(0, 172800)  # 2 days
    # V1-V28 are PCA components, normally distributed for benign
    v_features = {f"V{i}": np.random.normal(0, 1) for i in range(1, 29)}
    # Amount - typical transaction
    amount = random.choice([
        random.uniform(1, 50),      # Small purchase
        random.uniform(50, 200),    # Medium purchase
        random.uniform(200, 500),   # Larger purchase
    ])
    return {"Time": time, **v_features, "Amount": amount, "Class": 0}

def gen_fraudulent_transaction():
    """Generate fraudulent transaction (30 features)."""
    time = random.uniform(0, 172800)
    
    # Fraud patterns have different PCA distributions
    fraud_type = random.choice(['high_amount', 'unusual_pattern', 'rapid_sequence'])
    
    if fraud_type == 'high_amount':
        v_features = {f"V{i}": np.random.normal(0, 2) for i in range(1, 29)}
        v_features["V1"] = np.random.normal(-3, 1)  # Typical fraud signature
        v_features["V2"] = np.random.normal(2, 1)
        amount = random.uniform(500, 5000)
    elif fraud_type == 'unusual_pattern':
        v_features = {f"V{i}": np.random.normal(0, 1.5) for i in range(1, 29)}
        v_features["V3"] = np.random.normal(-4, 1)
        v_features["V4"] = np.random.normal(3, 1)
        v_features["V14"] = np.random.normal(-5, 1)  # Strong fraud indicator
        amount = random.uniform(100, 1000)
    else:  # rapid_sequence
        v_features = {f"V{i}": np.random.normal(0, 1.2) for i in range(1, 29)}
        v_features["V10"] = np.random.normal(-3, 1)
        v_features["V12"] = np.random.normal(-4, 1)
        v_features["V17"] = np.random.normal(-3, 1)
        amount = random.uniform(50, 300)
    
    return {"Time": time, **v_features, "Amount": amount, "Class": 1}

def generate_samples(n_benign: int, n_malicious: int, output_dir: Path, batch_size: int = 100000):
    """Generate and save samples."""
    total = n_benign + n_malicious
    tracker = ProgressTracker(total, "Fraud")
    
    output_path = output_dir / "fraud_expansion.jsonl"
    
    batch = []
    for _ in range(n_benign):
        batch.append(gen_benign_transaction())
        if len(batch) >= batch_size:
            append_to_jsonl(output_path, iter(batch))
            tracker.update(len(batch))
            batch = []
    
    for _ in range(n_malicious):
        batch.append(gen_fraudulent_transaction())
        if len(batch) >= batch_size:
            append_to_jsonl(output_path, iter(batch))
            tracker.update(len(batch))
            batch = []
    
    if batch:
        append_to_jsonl(output_path, iter(batch))
        tracker.update(len(batch))
    
    tracker.close()
    print(f"Saved: {output_path} ({total:,} samples)")

def main():
    parser = argparse.ArgumentParser(description="Generate fraud samples")
    parser.add_argument("--total", type=int, default=5_000_000, help="Total samples")
    parser.add_argument("--output", type=Path, default=Path("datasets/fraud_detection"))
    args = parser.parse_args()
    
    sampler = RatioSampler(args.total)
    n_benign, n_malicious = sampler.get_counts()
    print(f"Generating {n_benign:,} benign + {n_malicious:,} fraud = {args.total:,} total")
    
    generate_samples(n_benign, n_malicious, args.output)

if __name__ == "__main__":
    main()
