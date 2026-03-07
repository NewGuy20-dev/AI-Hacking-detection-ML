#!/usr/bin/env python3
from pathlib import Path


def _expand(globs):
    files = []
    for pattern in globs:
        files.extend(Path(".").glob(pattern))
    return [p for p in files if p.is_file()]


def _summarize(label, globs):
    files = _expand(globs)
    if not files:
        print(f"  [MISSING] {label}: NOT FOUND")
        return
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"  [OK] {label}: {len(files)} file(s), {total_bytes / (1024**3):.2f} GB")


print("Checking training dataset inputs (paths used by training scripts)...\n")

print("=== payload/url shared (benign + malicious corpora) ===")
_summarize("security_payloads (malicious)", [
    "datasets/security_payloads/**/*.txt",
    "datasets/security_payloads/**/*.jsonl",
])
_summarize("benign_60m", ["datasets/benign_60m/**/*.jsonl"])
_summarize("curated_benign", ["datasets/curated_benign/**/*.jsonl", "datasets/curated_benign/**/*.txt"])
_summarize("live_benign", ["datasets/live_benign/**/*.jsonl"])
_summarize("benign_5m.jsonl", ["datasets/benign_5m.jsonl"])
_summarize("fp_test_500k.jsonl", ["datasets/fp_test_500k.jsonl"])
print()

print("=== url model (train/val splits) ===")
_summarize("url train malicious", [
    "datasets/url_analysis/urlhaus.csv",
    "datasets/url_analysis/kaggle_malicious_urls.csv",
    "datasets/url_analysis/malicious_urls_5m.jsonl",
])
_summarize("url train benign", [
    "datasets/url_analysis/top-1m.csv",
    "datasets/live_benign/common_crawl_urls.jsonl",
])
_summarize("url val malicious", [
    "datasets/url_analysis/malicious_urls/malicious_phish.csv",
    "datasets/url_analysis/synthetic_malicious_hard.txt",
])
_summarize("url val benign", [
    "datasets/url_analysis/synthetic_benign_hard.txt",
])
print()

print("=== timeseries (LSTM) ===")
_summarize("timeseries benign", [
    "datasets/live_benign/timeseries_benign.npy",
    "datasets/timeseries/normal_traffic_expansion.npy",
])
_summarize("timeseries malicious/normal", [
    "datasets/timeseries/*.npy",
])
print()

print("=== sklearn models (network/fraud/host) ===")
_summarize("network intrusion", [
    "datasets/network_intrusion/synthetic_500k.jsonl",
    "datasets/live_benign/mawi_network_kdd.jsonl",
])
_summarize("fraud detection", [
    "datasets/fraud_detection/synthetic_500k.jsonl",
    "datasets/live_benign/fraud_benign.jsonl",
])
_summarize("host behavior", [
    "datasets/host_behavior/synthetic_500k.jsonl",
    "datasets/live_benign/host_behavior_benign.jsonl",
])
