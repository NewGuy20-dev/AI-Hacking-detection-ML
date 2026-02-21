#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

def check_jsonl(path):
    """Check JSONL files for label field"""
    issues = []
    total = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            total += 1
            try:
                obj = json.loads(line)
                if 'label' not in obj:
                    issues.append(f"Line {i}: missing 'label'")
                elif obj['label'] not in [0, 1]:
                    issues.append(f"Line {i}: invalid label {obj['label']}")
            except json.JSONDecodeError:
                issues.append(f"Line {i}: invalid JSON")
            if i >= 1000:  # Sample first 1000
                break
    return total, issues

def check_npy(path):
    """Check numpy files for labels"""
    data = np.load(path)
    unique = np.unique(data)
    if not np.all(np.isin(unique, [0, 1])):
        return len(data), [f"Invalid labels: {unique}"]
    return len(data), []

datasets = {
    'security_payloads': ['benign_payloads.jsonl', 'malicious_payloads.jsonl'],
    'url_analysis': ['benign_urls.jsonl', 'malicious_urls.jsonl'],
    'network_intrusion': ['benign_network.jsonl', 'malicious_network.jsonl'],
    'host_behavior': ['benign_host.jsonl', 'malicious_host.jsonl'],
    'timeseries': ['benign_timeseries_labels.npy', 'malicious_timeseries_labels.npy'],
    'fraud_detection': ['benign_fraud.jsonl', 'malicious_fraud.jsonl']
}

print("Checking dataset labels...\n")
for dataset, files in datasets.items():
    print(f"=== {dataset} ===")
    for fname in files:
        path = Path(f"./datasets/{dataset}/{fname}")
        if not path.exists():
            print(f"  {fname}: NOT FOUND")
            continue
        
        if fname.endswith('.npy'):
            count, issues = check_npy(path)
        else:
            count, issues = check_jsonl(path)
        
        status = "✓" if not issues else "✗"
        print(f"  {status} {fname}: {count} samples")
        for issue in issues[:5]:
            print(f"      {issue}")
        if len(issues) > 5:
            print(f"      ... and {len(issues)-5} more issues")
    print()
