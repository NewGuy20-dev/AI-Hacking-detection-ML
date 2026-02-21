#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

VALID_LABELS = {
    'network': {'benign', 'DoS', 'Probe', 'R2L', 'U2R'},
    'url': {'benign', 'phishing', 'malware', 'defacement', 'spam'},
    'payload': {'benign', 'sqli', 'xss', 'cmdi', 'path_traversal', 'ssti', 'xxe', 'ldap'},
    'fraud': {'legitimate', 'fraud'},
    'host': {'benign', 'spyware', 'ransomware', 'trojan', 'rootkit', 'backdoor'},
    'timeseries': {'normal', 'ddos', 'port_scan', 'exfiltration', 'c2', 'brute_force'}
}

REQUIRED_UNIVERSAL = ['sample_id', 'label', 'label_binary', 'label_method', 'source_dataset', 
                      'is_synthetic', 'date_created', 'split', 'confidence']

SPECIFIC_REQUIRED = {
    'network': ['features'],
    'url': ['url', 'date_verified', 'is_live'],
    'payload': ['payload', 'context', 'is_adversarial'],
    'fraud': ['verified_by', 'fraud_type', 'features'],
    'host': ['malware_family', 'features'],
    'timeseries': ['window_seconds', 'attack_start_offset', 'attack_end_offset', 'sequence']
}

def validate_sample(obj, dataset_type):
    errors = []
    
    # Universal checks
    for field in REQUIRED_UNIVERSAL:
        if field not in obj:
            errors.append(f"Missing required field: {field}")
    
    if 'label' in obj and obj['label'] not in VALID_LABELS[dataset_type]:
        errors.append(f"Invalid label: {obj['label']}")
    
    if 'label_binary' in obj and obj['label_binary'] not in [0, 1]:
        errors.append(f"Invalid label_binary: {obj['label_binary']}")
    
    # Check label consistency
    if 'label' in obj and 'label_binary' in obj:
        is_benign = obj['label'] in ['benign', 'legitimate', 'normal']
        if (is_benign and obj['label_binary'] != 0) or (not is_benign and obj['label_binary'] != 1):
            errors.append(f"Label mismatch: label={obj['label']}, label_binary={obj['label_binary']}")
    
    # Synthetic samples in test/holdout
    if obj.get('is_synthetic') and obj.get('split') in ['test', 'holdout']:
        errors.append("Synthetic sample in test/holdout split")
    
    # Type-specific checks
    if dataset_type in SPECIFIC_REQUIRED:
        for field in SPECIFIC_REQUIRED[dataset_type]:
            # Skip attack-specific fields for benign samples
            if field in ['attack_start_offset', 'attack_end_offset'] and obj.get('label_binary') == 0:
                continue
            if field == 'fraud_type' and obj.get('label') == 'legitimate':
                continue
            if field == 'malware_family' and obj.get('label') == 'benign':
                continue
            if field not in obj:
                errors.append(f"Missing {dataset_type}-specific field: {field}")
    
    # Timeseries attack offset check
    if dataset_type == 'timeseries' and obj.get('label_binary') == 1:
        if 'attack_start_offset' not in obj or 'attack_end_offset' not in obj:
            errors.append("Attack sample missing offset fields")
    
    return errors

def check_jsonl_file(path, dataset_type, sample_limit=None):
    stats = {'total': 0, 'valid': 0, 'errors': defaultdict(int)}
    error_samples = []
    
    with open(path) as f:
        for i, line in enumerate(f, 1):
            stats['total'] += 1
            try:
                obj = json.loads(line)
                errors = validate_sample(obj, dataset_type)
                if errors:
                    for err in errors:
                        stats['errors'][err] += 1
                    if len(error_samples) < 10:
                        error_samples.append((i, errors))
                else:
                    stats['valid'] += 1
            except json.JSONDecodeError:
                stats['errors']['Invalid JSON'] += 1
            
            if sample_limit and i >= sample_limit:
                break
    
    return stats, error_samples

def check_timeseries_labels(data_path, labels_path):
    data = np.load(data_path)
    labels = np.load(labels_path)
    
    if len(data) != len(labels):
        return {'error': f"Shape mismatch: data={len(data)}, labels={len(labels)}"}
    
    unique = np.unique(labels)
    if not np.all(np.isin(unique, [0, 1])):
        return {'error': f"Invalid label values: {unique}"}
    
    return {'total': len(labels), 'valid': len(labels), 'benign': np.sum(labels==0), 'malicious': np.sum(labels==1)}

datasets = [
    ('datasets/security_payloads', 'payload', ['benign_payloads.jsonl', 'malicious_payloads.jsonl']),
    ('datasets/url_analysis', 'url', ['benign_urls.jsonl', 'malicious_urls.jsonl']),
    ('datasets/network_intrusion', 'network', ['benign_network.jsonl', 'malicious_network.jsonl']),
    ('datasets/host_behavior', 'host', ['benign_host.jsonl', 'malicious_host.jsonl']),
    ('datasets/fraud_detection', 'fraud', ['benign_fraud.jsonl', 'malicious_fraud.jsonl']),
]

print("=" * 80)
print("DATASET LABEL VALIDATION REPORT")
print("=" * 80)
print()

total_samples = 0
total_valid = 0
total_invalid = 0

for dataset_path, dataset_type, files in datasets:
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_path}")
    print(f"Type: {dataset_type}")
    print(f"{'='*80}")
    
    for fname in files:
        path = Path(dataset_path) / fname
        if not path.exists():
            print(f"\n  ❌ {fname}: FILE NOT FOUND")
            continue
        
        print(f"\n  📄 {fname}")
        stats, error_samples = check_jsonl_file(path, dataset_type, sample_limit=100000)
        
        total_samples += stats['total']
        total_valid += stats['valid']
        total_invalid += (stats['total'] - stats['valid'])
        
        validity_pct = (stats['valid'] / stats['total'] * 100) if stats['total'] > 0 else 0
        status = "✅" if validity_pct == 100 else "⚠️" if validity_pct >= 95 else "❌"
        
        print(f"     {status} Samples: {stats['total']:,} | Valid: {stats['valid']:,} ({validity_pct:.2f}%)")
        
        if stats['errors']:
            print(f"\n     Error Summary:")
            for error, count in sorted(stats['errors'].items(), key=lambda x: -x[1])[:10]:
                print(f"       • {error}: {count:,} occurrences")
        
        if error_samples:
            print(f"\n     Sample Errors (first 3):")
            for line_num, errors in error_samples[:3]:
                print(f"       Line {line_num}: {errors[0]}")

# Check timeseries separately
print(f"\n{'='*80}")
print(f"Dataset: datasets/timeseries")
print(f"Type: timeseries")
print(f"{'='*80}")

ts_files = [
    ('benign_timeseries.npy', 'benign_timeseries_labels.npy'),
    ('malicious_timeseries.npy', 'malicious_timeseries_labels.npy')
]

for data_file, label_file in ts_files:
    data_path = Path('datasets/timeseries') / data_file
    label_path = Path('datasets/timeseries') / label_file
    
    if not data_path.exists() or not label_path.exists():
        print(f"\n  ❌ {data_file}: FILE NOT FOUND")
        continue
    
    print(f"\n  📄 {data_file}")
    result = check_timeseries_labels(data_path, label_path)
    
    if 'error' in result:
        print(f"     ❌ {result['error']}")
    else:
        total_samples += result['total']
        total_valid += result['valid']
        print(f"     ✅ Samples: {result['total']:,} | Benign: {result['benign']:,} | Malicious: {result['malicious']:,}")

print(f"\n{'='*80}")
print(f"OVERALL SUMMARY")
print(f"{'='*80}")
print(f"Total Samples Checked: {total_samples:,}")
print(f"Valid Samples: {total_valid:,} ({total_valid/total_samples*100:.2f}%)")
print(f"Invalid Samples: {total_invalid:,} ({total_invalid/total_samples*100:.2f}%)")
print()
