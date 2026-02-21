#!/usr/bin/env python3
import json
import numpy as np
import random
from pathlib import Path
from collections import Counter

def sample_jsonl(path, n=100):
    """Random sample from JSONL"""
    samples = []
    with open(path) as f:
        lines = f.readlines()
    for line in random.sample(lines, min(n, len(lines))):
        samples.append(json.loads(line))
    return samples

def check_url_quality(benign_path, malicious_path):
    print("\n" + "="*80)
    print("URL ANALYSIS - Label Quality Check")
    print("="*80)
    
    benign = sample_jsonl(benign_path, 50)
    malicious = sample_jsonl(malicious_path, 50)
    
    issues = []
    
    # Check benign URLs
    print("\n📊 Benign URL Samples (first 10):")
    for i, s in enumerate(benign[:10], 1):
        url = s.get('text', '')
        label = s.get('label', -1)
        print(f"  {i}. {url[:80]} | label={label}")
        
        if label != 0:
            issues.append(f"Benign file has label={label}: {url}")
        # Check for obvious malicious patterns in benign
        if any(x in url.lower() for x in ['phishing', 'malware', 'hack', 'exploit', 'payload']):
            issues.append(f"Suspicious benign URL: {url}")
    
    # Check malicious URLs
    print("\n📊 Malicious URL Samples (first 10):")
    for i, s in enumerate(malicious[:10], 1):
        url = s.get('text', '')
        label = s.get('label', -1)
        print(f"  {i}. {url[:80]} | label={label}")
        
        if label != 1:
            issues.append(f"Malicious file has label={label}: {url}")
    
    # Label distribution
    benign_labels = Counter([s.get('label') for s in benign])
    mal_labels = Counter([s.get('label') for s in malicious])
    
    print(f"\n📈 Label Distribution:")
    print(f"  Benign file: {dict(benign_labels)}")
    print(f"  Malicious file: {dict(mal_labels)}")
    
    return issues

def check_payload_quality(malicious_path):
    print("\n" + "="*80)
    print("PAYLOAD - Label Quality Check")
    print("="*80)
    
    malicious = sample_jsonl(malicious_path, 50)
    issues = []
    
    print("\n📊 Malicious Payload Samples (first 15):")
    for i, s in enumerate(malicious[:15], 1):
        payload = s.get('text', '')
        label = s.get('label', -1)
        print(f"  {i}. {payload[:60]} | label={label}")
        
        if label != 1:
            issues.append(f"Malicious file has label={label}: {payload}")
        
        # Check if it's actually malicious
        benign_patterns = ['hello', 'test', 'example', 'normal', 'valid']
        if payload.lower() in benign_patterns:
            issues.append(f"Possibly benign in malicious: {payload}")
    
    mal_labels = Counter([s.get('label') for s in malicious])
    print(f"\n📈 Label Distribution: {dict(mal_labels)}")
    
    return issues

def check_network_quality(path):
    print("\n" + "="*80)
    print("NETWORK INTRUSION - Label Quality Check")
    print("="*80)
    
    samples = sample_jsonl(path, 100)
    issues = []
    
    labels = Counter([s.get('label') for s in samples])
    print(f"\n📈 Label Distribution: {dict(labels)}")
    print(f"  Total sampled: {len(samples)}")
    print(f"  Benign (0): {labels[0]} ({labels[0]/len(samples)*100:.1f}%)")
    print(f"  Malicious (1): {labels[1]} ({labels[1]/len(samples)*100:.1f}%)")
    
    # Check for suspicious patterns
    benign = [s for s in samples if s.get('label') == 0]
    malicious = [s for s in samples if s.get('label') == 1]
    
    print(f"\n📊 Benign Sample Features (first 3):")
    for i, s in enumerate(benign[:3], 1):
        print(f"  {i}. duration={s.get('duration'):.2f}, src_bytes={s.get('src_bytes')}, dst_bytes={s.get('dst_bytes')}, protocol={s.get('protocol_type')}")
    
    print(f"\n📊 Malicious Sample Features (first 3):")
    for i, s in enumerate(malicious[:3], 1):
        print(f"  {i}. duration={s.get('duration'):.2f}, src_bytes={s.get('src_bytes')}, dst_bytes={s.get('dst_bytes')}, protocol={s.get('protocol_type')}")
    
    # Check for label consistency
    if labels[0] == 0 or labels[1] == 0:
        issues.append(f"Imbalanced: only one class present in sample")
    
    return issues

def check_host_quality(path):
    print("\n" + "="*80)
    print("HOST BEHAVIOR - Label Quality Check")
    print("="*80)
    
    samples = sample_jsonl(path, 100)
    issues = []
    
    labels = Counter([s.get('label') for s in samples])
    print(f"\n📈 Label Distribution: {dict(labels)}")
    print(f"  Total sampled: {len(samples)}")
    print(f"  Benign (0): {labels[0]} ({labels[0]/len(samples)*100:.1f}%)")
    print(f"  Malicious (1): {labels[1]} ({labels[1]/len(samples)*100:.1f}%)")
    
    benign = [s for s in samples if s.get('label') == 0]
    malicious = [s for s in samples if s.get('label') == 1]
    
    print(f"\n📊 Benign Sample (first 2):")
    for i, s in enumerate(benign[:2], 1):
        print(f"  {i}. nproc={s.get('pslist_nproc')}, ndlls={s.get('dlllist_ndlls')}, nhandles={s.get('handles_nhandles')}")
    
    print(f"\n📊 Malicious Sample (first 2):")
    for i, s in enumerate(malicious[:2], 1):
        print(f"  {i}. nproc={s.get('pslist_nproc')}, ndlls={s.get('dlllist_ndlls')}, nhandles={s.get('handles_nhandles')}")
    
    return issues

def check_fraud_quality(path):
    print("\n" + "="*80)
    print("FRAUD DETECTION - Label Quality Check")
    print("="*80)
    
    samples = sample_jsonl(path, 100)
    issues = []
    
    labels = Counter([s.get('Class') for s in samples])
    print(f"\n📈 Label Distribution: {dict(labels)}")
    print(f"  Total sampled: {len(samples)}")
    print(f"  Legitimate (0): {labels[0]} ({labels[0]/len(samples)*100:.1f}%)")
    print(f"  Fraud (1): {labels[1]} ({labels[1]/len(samples)*100:.1f}%)")
    
    # Check for missing labels
    for s in samples:
        if 'Class' not in s:
            issues.append("Sample missing 'Class' field")
    
    return issues

def check_timeseries_quality():
    print("\n" + "="*80)
    print("TIMESERIES - Label Quality Check")
    print("="*80)
    
    issues = []
    
    normal = np.load('./datasets/timeseries/normal_traffic_expansion.npy')
    attack = np.load('./datasets/timeseries/attack_traffic_expansion.npy')
    
    print(f"\n📊 Normal Traffic:")
    print(f"  Shape: {normal.shape}")
    print(f"  Min: {normal.min():.2f}, Max: {normal.max():.2f}, Mean: {normal.mean():.2f}")
    print(f"  Sample [0,0,:5]: {normal[0,0,:5]}")
    
    print(f"\n📊 Attack Traffic:")
    print(f"  Shape: {attack.shape}")
    print(f"  Min: {attack.min():.2f}, Max: {attack.max():.2f}, Mean: {attack.mean():.2f}")
    print(f"  Sample [0,0,:5]: {attack[0,0,:5]}")
    
    # Check if they're too similar
    if abs(normal.mean() - attack.mean()) < 1:
        issues.append("Normal and attack traffic have very similar means - possible labeling issue")
    
    # Check for NaN/Inf
    if np.isnan(normal).any() or np.isnan(attack).any():
        issues.append("NaN values detected")
    if np.isinf(normal).any() or np.isinf(attack).any():
        issues.append("Inf values detected")
    
    return issues

# Run all checks
random.seed(42)
all_issues = []

try:
    issues = check_url_quality(
        './datasets/url_analysis/url_benign_expansion.jsonl',
        './datasets/url_analysis/url_malicious_expansion.jsonl'
    )
    all_issues.extend(issues)
except Exception as e:
    print(f"❌ URL check failed: {e}")

try:
    issues = check_payload_quality('./datasets/security_payloads/payload_malicious_expansion.jsonl')
    all_issues.extend(issues)
except Exception as e:
    print(f"❌ Payload check failed: {e}")

try:
    issues = check_network_quality('./datasets/network_intrusion/network_expansion.jsonl')
    all_issues.extend(issues)
except Exception as e:
    print(f"❌ Network check failed: {e}")

try:
    issues = check_host_quality('./datasets/host_behavior/host_expansion.jsonl')
    all_issues.extend(issues)
except Exception as e:
    print(f"❌ Host check failed: {e}")

try:
    issues = check_fraud_quality('./datasets/fraud_detection/fraud_expansion.jsonl')
    all_issues.extend(issues)
except Exception as e:
    print(f"❌ Fraud check failed: {e}")

try:
    issues = check_timeseries_quality()
    all_issues.extend(issues)
except Exception as e:
    print(f"❌ Timeseries check failed: {e}")

# Summary
print("\n" + "="*80)
print("QUALITY SUMMARY")
print("="*80)
if all_issues:
    print(f"\n⚠️  Found {len(all_issues)} potential issues:")
    for issue in all_issues[:20]:
        print(f"  • {issue}")
    if len(all_issues) > 20:
        print(f"  ... and {len(all_issues)-20} more")
else:
    print("\n✅ No obvious quality issues detected in sampled data")
