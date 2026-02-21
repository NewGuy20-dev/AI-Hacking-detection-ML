#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import time

class QualityChecker:
    def __init__(self):
        self.stats = defaultdict(lambda: {'total': 0, 'label_dist': Counter(), 'errors': []})
    
    def check_jsonl(self, path, dataset_name, expected_label=None):
        print(f"\n{'='*80}")
        print(f"Checking: {path.name}")
        print(f"{'='*80}")
        
        stats = self.stats[dataset_name]
        start = time.time()
        local_count = 0
        local_labels = Counter()
        local_errors = []
        
        with open(path, 'rb') as f:
            for i, line in enumerate(f, 1):
                local_count += 1
                
                try:
                    obj = json.loads(line)
                    
                    # Get label (handle both 'label' and 'Class')
                    label = obj.get('label', obj.get('Class', -1))
                    local_labels[label] += 1
                    
                    # Check label validity
                    if label not in [0, 1]:
                        if len(local_errors) < 100:
                            local_errors.append(f"Line {i}: invalid label {label}")
                    
                    # Check expected label if specified
                    elif expected_label is not None and label != expected_label:
                        if len(local_errors) < 100:
                            local_errors.append(f"Line {i}: expected {expected_label}, got {label}")
                    
                except:
                    if len(local_errors) < 100:
                        local_errors.append(f"Line {i}: parse error")
                
                # Progress
                if i % 1_000_000 == 0:
                    elapsed = time.time() - start
                    rate = i / elapsed
                    print(f"  Progress: {i:,} samples | {rate:,.0f} samples/sec | {elapsed:.1f}s")
        
        stats['total'] += local_count
        stats['label_dist'].update(local_labels)
        stats['errors'].extend(local_errors)
        
        elapsed = time.time() - start
        print(f"\n✓ Completed: {local_count:,} samples in {elapsed:.1f}s ({local_count/elapsed:,.0f} samples/sec)")
        print(f"  Label distribution: {dict(local_labels)}")
        if local_errors:
            print(f"  ⚠️  Errors: {len(local_errors)}")
    
    def check_timeseries(self, normal_path, attack_path):
        print(f"\n{'='*80}")
        print(f"Checking: Timeseries")
        print(f"{'='*80}")
        
        stats = self.stats['timeseries']
        
        print(f"\n  Loading {normal_path.name}...")
        normal = np.load(normal_path)
        stats['total'] += len(normal)
        stats['label_dist'][0] = len(normal)
        
        print(f"    Shape: {normal.shape}")
        print(f"    Range: [{normal.min():.2f}, {normal.max():.2f}]")
        print(f"    Mean: {normal.mean():.2f}, Std: {normal.std():.2f}")
        
        if np.isnan(normal).any():
            stats['errors'].append("Normal data contains NaN")
        if np.isinf(normal).any():
            stats['errors'].append("Normal data contains Inf")
        
        print(f"\n  Loading {attack_path.name}...")
        attack = np.load(attack_path)
        stats['total'] += len(attack)
        stats['label_dist'][1] = len(attack)
        
        print(f"    Shape: {attack.shape}")
        print(f"    Range: [{attack.min():.2f}, {attack.max():.2f}]")
        print(f"    Mean: {attack.mean():.2f}, Std: {attack.std():.2f}")
        
        if np.isnan(attack).any():
            stats['errors'].append("Attack data contains NaN")
        if np.isinf(attack).any():
            stats['errors'].append("Attack data contains Inf")
        
        # Check if distinguishable
        if abs(normal.mean() - attack.mean()) < 0.1 * normal.std():
            stats['errors'].append("Normal and attack distributions too similar")
        
        print(f"\n✓ Total timeseries samples: {stats['total']:,}")
        print(f"  Label distribution: {dict(stats['label_dist'])}")

checker = QualityChecker()

# URL Analysis
url_base = Path('./datasets/url_analysis')
for f in ['url_benign_expansion.jsonl', 'malicious_urls_5m.jsonl']:
    if (url_base / f).exists():
        expected = 0 if 'benign' in f else 1
        checker.check_jsonl(url_base / f, 'url', expected_label=expected)

# Payload - check all txt files
payload_base = Path('./datasets/security_payloads')
if (payload_base / 'payload_malicious_expansion.jsonl').exists():
    checker.check_jsonl(payload_base / 'payload_malicious_expansion.jsonl', 'payload', expected_label=1)

for txt_file in payload_base.rglob('*.txt'):
    if txt_file.stat().st_size > 0:
        print(f"\n{'='*80}")
        print(f"Checking: {txt_file.relative_to(payload_base)}")
        print(f"{'='*80}")
        stats = checker.stats['payload']
        with open(txt_file, errors='ignore') as f:
            for i, line in enumerate(f, 1):
                stats['total'] += 1
                stats['label_dist'][1] += 1  # Payloads are malicious
                if i % 1_000_000 == 0:
                    print(f"  Progress: {i:,} lines")
        print(f"✓ Completed: {i:,} lines")

# Benign payloads
benign_base = Path('./datasets')
for f in ['benign_5m.jsonl']:
    if (benign_base / f).exists():
        checker.check_jsonl(benign_base / f, 'payload', expected_label=0)

for jsonl_file in (benign_base / 'curated_benign').rglob('*.jsonl'):
    if jsonl_file.stat().st_size > 0:
        checker.check_jsonl(jsonl_file, 'payload', expected_label=0)

for jsonl_file in (benign_base / 'live_benign').rglob('*.jsonl'):
    if jsonl_file.stat().st_size > 0:
        checker.check_jsonl(jsonl_file, 'payload', expected_label=0)

for jsonl_file in (benign_base / 'benign_60m').rglob('*.jsonl'):
    if jsonl_file.stat().st_size > 0:
        checker.check_jsonl(jsonl_file, 'payload', expected_label=0)

# Network
network_base = Path('./datasets/network_intrusion')
for f in ['network_expansion.jsonl', 'synthetic_500k.jsonl']:
    if (network_base / f).exists():
        checker.check_jsonl(network_base / f, 'network')

# Host
host_base = Path('./datasets/host_behavior')
for f in ['host_expansion.jsonl', 'synthetic_500k.jsonl']:
    if (host_base / f).exists():
        checker.check_jsonl(host_base / f, 'host')

# Fraud
fraud_base = Path('./datasets/fraud_detection')
for f in ['fraud_expansion.jsonl', 'synthetic_500k.jsonl']:
    if (fraud_base / f).exists():
        checker.check_jsonl(fraud_base / f, 'fraud')

# Timeseries
ts_base = Path('./datasets/timeseries')
for normal, attack in [
    ('normal_traffic_expansion.npy', 'attack_traffic_expansion.npy'),
    ('normal_traffic_500k.npy', 'attack_traffic_500k.npy'),
    ('normal_traffic_improved.npy', 'attack_traffic_improved.npy')
]:
    if (ts_base / normal).exists() and (ts_base / attack).exists():
        checker.check_timeseries(ts_base / normal, ts_base / attack)

# Final Summary
print(f"\n{'='*80}")
print("FINAL QUALITY REPORT")
print(f"{'='*80}\n")

total_samples = 0
total_errors = 0

for dataset, stats in checker.stats.items():
    total_samples += stats['total']
    total_errors += len(stats['errors'])
    
    error_rate = len(stats['errors']) / stats['total'] * 100 if stats['total'] > 0 else 0
    status = "✅" if error_rate == 0 else "⚠️" if error_rate < 1 else "❌"
    
    print(f"{status} {dataset.upper()}")
    print(f"   Samples: {stats['total']:,}")
    print(f"   Labels: {dict(stats['label_dist'])}")
    print(f"   Errors: {len(stats['errors'])} ({error_rate:.4f}%)")
    
    if stats['errors']:
        print(f"   First 5 errors:")
        for err in stats['errors'][:5]:
            print(f"     • {err}")
    print()

print(f"{'='*80}")
print(f"TOTAL: {total_samples:,} samples checked")
print(f"ERRORS: {total_errors:,} ({total_errors/total_samples*100:.4f}%)")
print(f"{'='*80}")

if total_errors == 0:
    print("\n✅ All labels are valid!")
elif total_errors / total_samples < 0.01:
    print(f"\n⚠️  Minor issues found ({total_errors/total_samples*100:.4f}% error rate)")
else:
    print(f"\n❌ Significant labeling issues detected ({total_errors/total_samples*100:.4f}% error rate)")
