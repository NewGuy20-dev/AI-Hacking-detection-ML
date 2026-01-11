#!/usr/bin/env python3
"""Generate realistic benign fraud/transaction samples (PCA format)."""

import json
import random
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "fraud_benign.jsonl"
TARGET_SAMPLES = 5_000_000

PROFILES = {
    'small': {'amount': (1, 50), 'v_scale': 2},
    'medium': {'amount': (50, 200), 'v_scale': 1.5},
    'large': {'amount': (200, 1000), 'v_scale': 1},
    'recurring': {'amount': (10, 150), 'v_scale': 0.5},
    'grocery': {'amount': (20, 300), 'v_scale': 1},
}

def generate_sample(profile_name: str, time_offset: float) -> dict:
    p = PROFILES[profile_name]
    scale = p['v_scale']
    
    return {
        'Time': time_offset,
        'V1': round(random.gauss(0, 1.5 * scale), 6), 'V2': round(random.gauss(0, 1.2), 6),
        'V3': round(random.gauss(0, 1.0), 6), 'V4': round(random.gauss(0, 1.2), 6),
        'V5': round(random.gauss(0, 0.8), 6), 'V6': round(random.gauss(0, 0.8), 6),
        'V7': round(random.gauss(0, 0.7), 6), 'V8': round(random.gauss(0, 0.5), 6),
        'V9': round(random.gauss(0, 0.6), 6), 'V10': round(random.gauss(0, 0.8), 6),
        'V11': round(random.gauss(0, 0.7), 6), 'V12': round(random.gauss(0, 0.6), 6),
        'V13': round(random.gauss(0, 0.5), 6), 'V14': round(random.gauss(0, 1.0 * scale), 6),
        'V15': round(random.gauss(0, 0.4), 6), 'V16': round(random.gauss(0, 0.5), 6),
        'V17': round(random.gauss(0, 0.6), 6), 'V18': round(random.gauss(0, 0.4), 6),
        'V19': round(random.gauss(0, 0.3), 6), 'V20': round(random.gauss(0, 0.3), 6),
        'V21': round(random.gauss(0, 0.3), 6), 'V22': round(random.gauss(0, 0.3), 6),
        'V23': round(random.gauss(0, 0.2), 6), 'V24': round(random.gauss(0, 0.2), 6),
        'V25': round(random.gauss(0, 0.2), 6), 'V26': round(random.gauss(0, 0.2), 6),
        'V27': round(random.gauss(0, 0.1), 6), 'V28': round(random.gauss(0, 0.1), 6),
        'Amount': round(random.uniform(*p['amount']), 2), 'Class': 0
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_SAMPLES:,} samples")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE, 'r', encoding='utf-8'))
        if existing >= TARGET_SAMPLES:
            print(f"Already complete with {existing:,} samples!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")
    
    profiles = list(PROFILES.keys())
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for i in tqdm(range(TARGET_SAMPLES), desc="Generating"):
            f.write(json.dumps(generate_sample(random.choice(profiles), random.uniform(0, 172800))) + '\n')
    
    print(f"\nGenerated {TARGET_SAMPLES:,} samples -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
