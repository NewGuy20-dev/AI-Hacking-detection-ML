#!/usr/bin/env python3
"""Count total data samples across all datasets."""
import json
from pathlib import Path

def count_jsonl(file):
    """Count lines in JSONL file."""
    try:
        return sum(1 for _ in open(file, 'r', encoding='utf-8', errors='ignore'))
    except:
        return 0

def count_csv(file):
    """Count lines in CSV file (minus header)."""
    try:
        count = sum(1 for _ in open(file, 'r', encoding='utf-8', errors='ignore'))
        return max(0, count - 1)  # Subtract header
    except:
        return 0

def count_npy(file):
    """Count samples in numpy file."""
    try:
        import numpy as np
        return len(np.load(file, mmap_mode='r'))
    except:
        return 0

def count_txt(file):
    """Count lines in text file."""
    try:
        return sum(1 for _ in open(file, 'r', encoding='utf-8', errors='ignore'))
    except:
        return 0

datasets_dir = Path('datasets')

# Define dataset files
datasets = {
    'Payload (Malicious)': [
        'security_payloads/**/*.txt',
    ],
    'Payload (Benign)': [
        'curated_benign/**/*.txt',
        'live_benign/wikipedia_text.jsonl',
        'live_benign/github_snippets.jsonl',
        'live_benign/stackoverflow_posts.jsonl',
        'benign_5m.jsonl',
        'benign_60m/**/*.jsonl',
    ],
    'URL (Malicious)': [
        'url_analysis/urlhaus.csv',
        'url_analysis/kaggle_malicious_urls.csv',
        'url_analysis/malicious_urls_5m.jsonl',
        'malicious_urls/synthetic_malicious_urls.jsonl',
    ],
    'URL (Benign)': [
        'url_analysis/top-1m.csv',
        'live_benign/common_crawl_urls.jsonl',
    ],
    'Network (Malicious)': [
        'network_intrusion/synthetic_500k.jsonl',
    ],
    'Network (Benign)': [
        'live_benign/mawi_network_kdd.jsonl',
    ],
    'Fraud (Malicious)': [
        'fraud_detection/synthetic_500k.jsonl',
    ],
    'Fraud (Benign)': [
        'fraud_detection/creditcard.csv',
        'live_benign/fraud_benign.jsonl',
    ],
    'Host (Malicious)': [
        'host_behavior/synthetic_500k.jsonl',
        'cic_malmem_full/MalMem2022.csv',
    ],
    'Host (Benign)': [
        'live_benign/host_behavior_benign.jsonl',
    ],
    'Timeseries (Malicious)': [
        'timeseries/attack_traffic_500k.npy',
    ],
    'Timeseries (Benign)': [
        'timeseries/normal_traffic_500k.npy',
    ],
}

print("=" * 80)
print("DATASET SAMPLE COUNT")
print("=" * 80)

total = 0
category_totals = {}

for category, patterns in datasets.items():
    count = 0
    for pattern in patterns:
        for file in datasets_dir.glob(pattern):
            if file.is_file():
                if file.suffix == '.jsonl':
                    count += count_jsonl(file)
                elif file.suffix == '.csv':
                    count += count_csv(file)
                elif file.suffix == '.npy':
                    count += count_npy(file)
                elif file.suffix == '.txt':
                    count += count_txt(file)
    
    category_totals[category] = count
    total += count
    print(f"{category:30s}: {count:>15,}")

print("=" * 80)
print(f"{'TOTAL':30s}: {total:>15,}")
print("=" * 80)

# Summary by model
print("\nSUMMARY BY MODEL:")
print("-" * 80)
models = {
    'Payload CNN': category_totals.get('Payload (Malicious)', 0) + category_totals.get('Payload (Benign)', 0),
    'URL CNN': category_totals.get('URL (Malicious)', 0) + category_totals.get('URL (Benign)', 0),
    'Network RF': category_totals.get('Network (Malicious)', 0) + category_totals.get('Network (Benign)', 0),
    'Fraud XGBoost': category_totals.get('Fraud (Malicious)', 0) + category_totals.get('Fraud (Benign)', 0),
    'Host RF': category_totals.get('Host (Malicious)', 0) + category_totals.get('Host (Benign)', 0),
    'Timeseries LSTM': category_totals.get('Timeseries (Malicious)', 0) + category_totals.get('Timeseries (Benign)', 0),
}

for model, count in models.items():
    print(f"{model:30s}: {count:>15,}")
print("-" * 80)
