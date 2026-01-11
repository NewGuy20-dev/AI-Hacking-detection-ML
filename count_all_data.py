#!/usr/bin/env python3
"""Count all data samples across all datasets."""
import os
import json
import gzip
import numpy as np
from pathlib import Path

def count_jsonl(file_path):
    """Count lines in JSONL file."""
    try:
        count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                count += 1
        return count
    except:
        return 0

def count_csv(file_path):
    """Count lines in CSV file (minus header)."""
    try:
        count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                count += 1
        return max(0, count - 1)  # Subtract header
    except:
        return 0

def count_npy(file_path):
    """Count samples in numpy file."""
    try:
        data = np.load(file_path, allow_pickle=True)
        return len(data)
    except:
        return 0

def count_txt(file_path):
    """Count lines in text file."""
    try:
        count = 0
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    except:
        return 0

def count_gz(file_path):
    """Count lines in gzipped file."""
    try:
        count = 0
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for _ in f:
                count += 1
        return count
    except:
        return 0

def count_directory_files(dir_path):
    """Count individual files in directory (like email_spam/easy_ham)."""
    try:
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        # Exclude metadata files
        files = [f for f in files if not f.startswith('.') and f != 'cmds']
        return len(files)
    except:
        return 0

def main():
    base = Path(r"D:\Vibe- Coding projects\AI-Hacking-detection-ML\datasets")
    
    categories = {}
    total = 0
    
    print("=" * 80)
    print("DATASET SAMPLE COUNT ANALYSIS")
    print("=" * 80)
    
    # 1. Network Intrusion
    print("\n[1] NETWORK INTRUSION")
    net_count = 0
    
    # Synthetic
    synthetic_net = base / "network_intrusion/synthetic_500k.jsonl"
    if synthetic_net.exists():
        c = count_jsonl(synthetic_net)
        print(f"  - Synthetic 500k: {c:,}")
        net_count += c
    
    # NSL-KDD
    nsl_train = base / "network_intrusion/nsl_kdd/KDDTrain+.txt"
    nsl_test = base / "network_intrusion/nsl_kdd/KDDTest+.txt"
    if nsl_train.exists():
        c = count_txt(nsl_train)
        print(f"  - NSL-KDD Train: {c:,}")
        net_count += c
    if nsl_test.exists():
        c = count_txt(nsl_test)
        print(f"  - NSL-KDD Test: {c:,}")
        net_count += c
    
    # KDD99
    kdd99_data = base / "network_intrusion/kdd99/kddcup.data.gz"
    if kdd99_data.exists():
        c = count_gz(kdd99_data)
        print(f"  - KDD99 Full: {c:,}")
        net_count += c
    
    # MAWI benign
    mawi = base / "live_benign/mawi_network_kdd.jsonl"
    if mawi.exists():
        c = count_jsonl(mawi)
        print(f"  - MAWI Benign: {c:,}")
        net_count += c
    
    print(f"  TOTAL: {net_count:,}")
    categories['Network Intrusion'] = net_count
    total += net_count
    
    # 2. URL Analysis
    print("\n[2] URL ANALYSIS")
    url_count = 0
    
    # Malicious URLs
    mal_5m = base / "url_analysis/malicious_urls_5m.jsonl"
    if mal_5m.exists():
        c = count_jsonl(mal_5m)
        print(f"  - Malicious 5M: {c:,}")
        url_count += c
    
    kaggle_mal = base / "url_analysis/kaggle_malicious_urls.csv"
    if kaggle_mal.exists():
        c = count_csv(kaggle_mal)
        print(f"  - Kaggle Malicious: {c:,}")
        url_count += c
    
    urlhaus = base / "url_analysis/urlhaus.csv"
    if urlhaus.exists():
        c = count_csv(urlhaus)
        print(f"  - URLhaus: {c:,}")
        url_count += c
    
    # Benign URLs
    tranco = base / "url_analysis/top-1m.csv"
    if tranco.exists():
        c = count_csv(tranco)
        print(f"  - Tranco Top 1M: {c:,}")
        url_count += c
    
    common_crawl = base / "live_benign/common_crawl_urls.jsonl"
    if common_crawl.exists():
        c = count_jsonl(common_crawl)
        print(f"  - Common Crawl URLs: {c:,}")
        url_count += c
    
    print(f"  TOTAL: {url_count:,}")
    categories['URL Analysis'] = url_count
    total += url_count
    
    # 3. Fraud Detection
    print("\n[3] FRAUD DETECTION")
    fraud_count = 0
    
    synthetic_fraud = base / "fraud_detection/synthetic_500k.jsonl"
    if synthetic_fraud.exists():
        c = count_jsonl(synthetic_fraud)
        print(f"  - Synthetic 500k: {c:,}")
        fraud_count += c
    
    creditcard = base / "fraud_detection/creditcard.csv"
    if creditcard.exists():
        c = count_csv(creditcard)
        print(f"  - Credit Card Dataset: {c:,}")
        fraud_count += c
    
    fraud_benign = base / "live_benign/fraud_benign.jsonl"
    if fraud_benign.exists():
        c = count_jsonl(fraud_benign)
        print(f"  - Fraud Benign: {c:,}")
        fraud_count += c
    
    print(f"  TOTAL: {fraud_count:,}")
    categories['Fraud Detection'] = fraud_count
    total += fraud_count
    
    # 4. Timeseries
    print("\n[4] TIMESERIES")
    ts_count = 0
    
    normal_ts = base / "timeseries/normal_traffic_500k.npy"
    attack_ts = base / "timeseries/attack_traffic_500k.npy"
    if normal_ts.exists():
        c = count_npy(normal_ts)
        print(f"  - Normal Traffic 500k: {c:,}")
        ts_count += c
    if attack_ts.exists():
        c = count_npy(attack_ts)
        print(f"  - Attack Traffic 500k: {c:,}")
        ts_count += c
    
    ts_benign = base / "live_benign/timeseries_benign.npy"
    if ts_benign.exists():
        c = count_npy(ts_benign)
        print(f"  - Timeseries Benign: {c:,}")
        ts_count += c
    
    print(f"  TOTAL: {ts_count:,}")
    categories['Timeseries'] = ts_count
    total += ts_count
    
    # 5. Host Behavior
    print("\n[5] HOST BEHAVIOR")
    host_count = 0
    
    synthetic_host = base / "host_behavior/synthetic_500k.jsonl"
    if synthetic_host.exists():
        c = count_jsonl(synthetic_host)
        print(f"  - Synthetic 500k: {c:,}")
        host_count += c
    
    host_benign = base / "live_benign/host_behavior_benign.jsonl"
    if host_benign.exists():
        c = count_jsonl(host_benign)
        print(f"  - Host Behavior Benign: {c:,}")
        host_count += c
    
    print(f"  TOTAL: {host_count:,}")
    categories['Host Behavior'] = host_count
    total += host_count
    
    # 6. Curated Benign
    print("\n[6] CURATED BENIGN")
    benign_count = 0
    
    benign_5m = base / "benign_5m.jsonl"
    if benign_5m.exists():
        c = count_jsonl(benign_5m)
        print(f"  - Benign 5M: {c:,}")
        benign_count += c
    
    # Benign 60M
    benign_60m_dir = base / "benign_60m"
    if benign_60m_dir.exists():
        for file in benign_60m_dir.glob("*.jsonl"):
            c = count_jsonl(file)
            print(f"  - {file.name}: {c:,}")
            benign_count += c
    
    # FP test
    fp_test = base / "fp_test_500k.jsonl"
    if fp_test.exists():
        c = count_jsonl(fp_test)
        print(f"  - FP Test 500k: {c:,}")
        benign_count += c
    
    print(f"  TOTAL: {benign_count:,}")
    categories['Curated Benign'] = benign_count
    total += benign_count
    
    # 7. Live Benign (Wikipedia, GitHub, etc.)
    print("\n[7] LIVE BENIGN DATA")
    live_count = 0
    
    wikipedia = base / "live_benign/wikipedia_text.jsonl"
    if wikipedia.exists():
        c = count_jsonl(wikipedia)
        print(f"  - Wikipedia: {c:,}")
        live_count += c
    
    github = base / "live_benign/github_snippets.jsonl"
    if github.exists():
        c = count_jsonl(github)
        print(f"  - GitHub Snippets: {c:,}")
        live_count += c
    
    stackoverflow = base / "live_benign/stackoverflow_posts.jsonl"
    if stackoverflow.exists():
        c = count_jsonl(stackoverflow)
        print(f"  - StackOverflow: {c:,}")
        live_count += c
    
    enron = base / "live_benign/enron_emails.jsonl"
    if enron.exists():
        c = count_jsonl(enron)
        print(f"  - Enron Emails: {c:,}")
        live_count += c
    
    reddit = base / "live_benign/reddit_comments.jsonl"
    if reddit.exists():
        c = count_jsonl(reddit)
        print(f"  - Reddit Comments: {c:,}")
        live_count += c
    
    print(f"  TOTAL: {live_count:,}")
    categories['Live Benign'] = live_count
    total += live_count
    
    # 8. Email Spam
    print("\n[8] EMAIL SPAM")
    spam_count = 0
    
    easy_ham = base / "email_spam/easy_ham"
    if easy_ham.exists():
        c = count_directory_files(easy_ham)
        print(f"  - Easy Ham Emails: {c:,}")
        spam_count += c
    
    print(f"  TOTAL: {spam_count:,}")
    categories['Email Spam'] = spam_count
    total += spam_count
    
    # 9. Security Payloads
    print("\n[9] SECURITY PAYLOADS")
    payload_count = 0
    
    # Count all .txt files in security_payloads
    security_dir = base / "security_payloads"
    if security_dir.exists():
        for txt_file in security_dir.rglob("*.txt"):
            try:
                c = count_txt(txt_file)
                if c > 0:
                    payload_count += c
            except:
                pass
        print(f"  - Attack Patterns/Wordlists: {payload_count:,}")
    
    # Curated benign adversarial
    adversarial_dir = base / "curated_benign/adversarial"
    if adversarial_dir.exists():
        for txt_file in adversarial_dir.glob("*.txt"):
            c = count_txt(txt_file)
            payload_count += c
        print(f"  - Adversarial Benign Patterns: (included above)")
    
    print(f"  TOTAL: {payload_count:,}")
    categories['Security Payloads'] = payload_count
    total += payload_count
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY BY CATEGORY")
    print("=" * 80)
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total * 100) if total > 0 else 0
        print(f"{cat:30s}: {count:15,} ({pct:5.1f}%)")
    
    print("\n" + "=" * 80)
    print(f"TOTAL SAMPLES: {total:,}")
    print("=" * 80)
    
    # Benign vs Malicious breakdown
    print("\n" + "=" * 80)
    print("BENIGN vs MALICIOUS BREAKDOWN")
    print("=" * 80)
    
    benign_total = (
        categories.get('Live Benign', 0) +
        categories.get('Curated Benign', 0) +
        categories.get('Email Spam', 0) +
        1_000_000 +  # MAWI benign from network intrusion
        10_000_000 + # Common Crawl URLs (benign)
        999_999 +    # Tranco top 1M (benign)
        5_000_000 +  # Fraud benign
        1_000_000 +  # Timeseries benign
        5_000_000    # Host behavior benign
    )
    
    malicious_total = (
        (categories.get('Network Intrusion', 0) - 1_000_000) +  # Subtract MAWI benign
        (categories.get('URL Analysis', 0) - 10_000_000 - 999_999) +  # Subtract benign URLs
        (categories.get('Fraud Detection', 0) - 5_000_000) +  # Subtract fraud benign
        (categories.get('Timeseries', 0) - 1_000_000) +  # Subtract timeseries benign
        (categories.get('Host Behavior', 0) - 5_000_000) +  # Subtract host benign
        categories.get('Security Payloads', 0)  # All malicious
    )
    
    benign_pct = (benign_total / total * 100) if total > 0 else 0
    malicious_pct = (malicious_total / total * 100) if total > 0 else 0
    
    print(f"Benign Samples    : {benign_total:15,} ({benign_pct:5.1f}%)")
    print(f"Malicious Samples : {malicious_total:15,} ({malicious_pct:5.1f}%)")
    print(f"Ratio (B:M)       : {benign_total/malicious_total:.2f}:1")
    print("=" * 80)
    
    # Size info
    try:
        total_size = 0
        for f in base.rglob('*'):
            try:
                if f.is_file():
                    total_size += f.stat().st_size
            except:
                pass
        print(f"\nDataset directory size: ~{total_size / (1024**3):.2f} GB")
    except:
        print("\nDataset directory size: Unable to calculate")

if __name__ == "__main__":
    main()
