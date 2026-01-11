#!/usr/bin/env python3
"""Generate 500k synthetic fraud transactions based on real patterns."""
import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

def load_real_fraud_stats(csv_path):
    """Load statistics from real fraud data."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        fraud = df[df['Class'] == 1]
        normal = df[df['Class'] == 0]
        
        # Get V1-V28 statistics for fraud and normal
        stats = {'fraud': {}, 'normal': {}}
        for col in [f'V{i}' for i in range(1, 29)] + ['Amount']:
            if col in fraud.columns:
                stats['fraud'][col] = {'mean': fraud[col].mean(), 'std': fraud[col].std()}
                stats['normal'][col] = {'mean': normal[col].mean(), 'std': normal[col].std()}
        
        stats['fraud']['Time'] = {'mean': fraud['Time'].mean(), 'std': fraud['Time'].std()}
        stats['normal']['Time'] = {'mean': normal['Time'].mean(), 'std': normal['Time'].std()}
        
        return stats
    except Exception as e:
        print(f"Could not load real data: {e}, using default stats")
        return None

def generate_fraud_transaction(stats=None):
    """Generate synthetic fraud transaction."""
    if stats and 'fraud' in stats:
        s = stats['fraud']
        return {
            'Time': max(0, np.random.normal(s.get('Time', {}).get('mean', 50000), s.get('Time', {}).get('std', 30000))),
            **{f'V{i}': np.random.normal(s.get(f'V{i}', {}).get('mean', 0), max(0.1, s.get(f'V{i}', {}).get('std', 2))) for i in range(1, 29)},
            'Amount': max(0, np.random.exponential(150)),  # Fraud often small amounts
            'Class': 1
        }
    else:
        # Default fraud patterns (anomalous values)
        return {
            'Time': random.uniform(0, 172800),
            'V1': np.random.normal(-3, 2),   # Typically negative for fraud
            'V2': np.random.normal(2, 1.5),
            'V3': np.random.normal(-4, 2),
            'V4': np.random.normal(3, 1.5),
            'V5': np.random.normal(-1, 1),
            'V6': np.random.normal(-1, 1),
            'V7': np.random.normal(-3, 2),
            'V8': np.random.normal(0.5, 1),
            'V9': np.random.normal(-2, 1.5),
            'V10': np.random.normal(-4, 2),
            'V11': np.random.normal(2, 1.5),
            'V12': np.random.normal(-5, 2),
            'V13': np.random.normal(0, 1),
            'V14': np.random.normal(-6, 2),  # Strong fraud indicator
            'V15': np.random.normal(0, 1),
            'V16': np.random.normal(-4, 2),
            'V17': np.random.normal(-5, 2),
            'V18': np.random.normal(-2, 1.5),
            'V19': np.random.normal(0.5, 1),
            'V20': np.random.normal(0.2, 0.5),
            'V21': np.random.normal(0.5, 0.5),
            'V22': np.random.normal(0, 1),
            'V23': np.random.normal(-0.2, 0.5),
            'V24': np.random.normal(0, 0.5),
            'V25': np.random.normal(0.2, 0.5),
            'V26': np.random.normal(0, 0.5),
            'V27': np.random.normal(0.5, 0.5),
            'V28': np.random.normal(0.2, 0.5),
            'Amount': max(0, np.random.exponential(100)),
            'Class': 1
        }

def generate_normal_transaction(stats=None):
    """Generate synthetic normal transaction."""
    if stats and 'normal' in stats:
        s = stats['normal']
        return {
            'Time': max(0, np.random.normal(s.get('Time', {}).get('mean', 90000), s.get('Time', {}).get('std', 50000))),
            **{f'V{i}': np.random.normal(s.get(f'V{i}', {}).get('mean', 0), max(0.1, s.get(f'V{i}', {}).get('std', 1))) for i in range(1, 29)},
            'Amount': max(0, np.random.exponential(80)),
            'Class': 0
        }
    else:
        # Default normal patterns (centered around 0)
        return {
            'Time': random.uniform(0, 172800),
            **{f'V{i}': np.random.normal(0, 1) for i in range(1, 29)},
            'Amount': max(0, np.random.exponential(80)),
            'Class': 0
        }

def main():
    base = Path(__file__).parent.parent
    output_path = base / "datasets" / "fraud_detection" / "synthetic_500k.jsonl"
    real_data_path = base / "datasets" / "fraud_detection" / "creditcard.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to load real data statistics
    stats = None
    if real_data_path.exists():
        print("Loading statistics from real fraud data...")
        stats = load_real_fraud_stats(real_data_path)
    
    total = 500_000
    fraud_count = 250_000  # 50% fraud for balanced training
    normal_count = 250_000
    
    print(f"Generating {total:,} fraud detection samples...")
    print(f"Output: {output_path}")
    
    samples = []
    
    # Generate fraud samples
    for _ in tqdm(range(fraud_count), desc="Fraud"):
        samples.append(generate_fraud_transaction(stats))
    
    # Generate normal samples
    for _ in tqdm(range(normal_count), desc="Normal"):
        samples.append(generate_normal_transaction(stats))
    
    random.shuffle(samples)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            # Round floats for smaller file size
            rounded = {k: round(v, 6) if isinstance(v, float) else v for k, v in sample.items()}
            f.write(json.dumps(rounded) + '\n')
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Generated {total:,} samples ({size_mb:.1f} MB)")
    print(f"  Fraud: {fraud_count:,}, Normal: {normal_count:,}")

if __name__ == "__main__":
    main()
