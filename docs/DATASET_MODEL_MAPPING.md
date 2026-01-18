# Dataset to Model Mapping

This document categorizes all datasets by which model they should train.

## Summary Table

| Model | Type | Malicious Data | Benign Data | Est. Samples |
|-------|------|----------------|-------------|--------------|
| **Payload CNN** | PyTorch | security_payloads/* | benign_60m/*, curated_benign/* | ~100M |
| **URL CNN** | PyTorch | url_analysis/malicious* | url_analysis/top-1m, benign_60m/urls_25m | ~30M |
| **Network Intrusion** | sklearn RF | network_intrusion/* | network_intrusion/* (normal) | ~500k |
| **Fraud Detection** | sklearn XGBoost | fraud_detection/creditcard.csv | fraud_detection/creditcard.csv | ~285k |
| **Timeseries LSTM** | PyTorch | timeseries/attack_traffic* | timeseries/normal_traffic* | ~30k |
| **Host Behavior** | sklearn | cic_malmem/* | cic_malmem/* (benign) | ~60k |

---

## 1. PAYLOAD CNN (Text/Injection Detection)

**Purpose**: Detect SQL injection, XSS, command injection, and other text-based attacks

### Malicious Data (Label=1)
```
datasets/security_payloads/
├── injection/
│   ├── sqlifuzzer/              # SQL injection payloads
│   ├── Tiny-XSS-Payloads/       # XSS payloads
│   ├── CRLF-Injection-Payloads/ # CRLF injection
│   └── Open-Redirect-Payloads/  # Open redirect
├── fuzzing/
│   ├── fuzzdb/                  # Fuzzing payloads
│   ├── IntruderPayloads/        # Burp intruder payloads
│   └── big-list-of-naughty-strings/
├── PayloadsAllTheThings/
│   ├── SQL Injection/           # SQL injection
│   ├── XSS Injection/           # XSS
│   ├── Command Injection/       # OS command injection
│   ├── Server Side Template Injection/
│   ├── LDAP Injection/
│   ├── NoSQL Injection/
│   ├── XPATH Injection/
│   ├── XXE Injection/
│   └── ... (all injection types)
├── SecLists/Fuzzing/            # SecLists fuzzing payloads
├── misc/                        # Misc attack payloads
└── passwords/                   # NOT for payload model (credential stuffing)
```

### Benign Data (Label=0)
```
datasets/benign_60m/
├── api_8m.jsonl      # 8M API requests/responses
├── code_6m.jsonl     # 6M code snippets (looks like code but benign)
├── configs_2m.jsonl  # 2M config file contents
├── logs_4m.jsonl     # 4M log entries
├── shell_5m.jsonl    # 5M shell commands (benign)
├── sql_8m.jsonl      # 8M legitimate SQL queries
└── text_2m.jsonl     # 2M natural text

datasets/curated_benign/
├── sentences.txt     # Natural sentences
├── names.txt         # Person names
├── emails.txt        # Email addresses
├── addresses.txt     # Physical addresses
├── products.txt      # Product descriptions
├── comments.txt      # User comments
├── json.txt          # JSON snippets
├── dates.txt         # Date strings
├── phones.txt        # Phone numbers
├── usernames.txt     # Usernames
├── search_queries.txt
└── adversarial/      # Hard negatives (look suspicious but benign)
    ├── sql_benign.txt        # "SELECT * FROM menu" type
    ├── code_benign.txt       # Code that looks like attacks
    ├── html_educational.txt  # HTML tutorials
    ├── command_tutorials.txt # Shell tutorials
    ├── heart_emoji.txt       # "<3" patterns
    ├── names_apostrophe.txt  # "O'Brien" names
    ├── menu_sql_like.txt     # Restaurant menus
    └── ...

datasets/benign_5m.jsonl        # 5M mixed benign samples
datasets/fp_test_500k.jsonl     # 500k FP test samples (DO NOT TRAIN)
```

### NOT for Payload Model
- `passwords/` - These are credential lists, not injection payloads
- `wordlists/` - Directory/subdomain bruteforce lists, not payloads

---

## 2. URL CNN (Malicious URL Detection)

**Purpose**: Detect phishing, malware, and malicious URLs

### Malicious Data (Label=1)
```
datasets/url_analysis/
├── kaggle_malicious_urls.csv   # 194k malicious URLs
├── urlhaus.csv                 # URLhaus malware URLs
├── real_malicious_urls.txt     # Real-world malicious
├── synthetic_malicious_hard.txt # Hard synthetic malicious
├── malicious_urls/             # Additional malicious URL lists
└── phishing/                   # Phishing URLs
```

### Benign Data (Label=0)
```
datasets/url_analysis/
├── top-1m.csv                  # Tranco top 1M domains
├── tranco_top1m.csv.zip        # Tranco compressed
├── synthetic_benign_hard.txt   # Hard synthetic benign
└── domains/                    # Legitimate domain lists

datasets/benign_60m/
└── urls_25m.jsonl              # 25M benign URLs
```

---

## 3. NETWORK INTRUSION (RandomForest)

**Purpose**: Detect network-based attacks (DoS, Probe, R2L, U2R)

### Data (Mixed labels in CSV)
```
datasets/network_intrusion/
├── nsl_kdd/
│   ├── KDDTrain+.txt          # Training data
│   └── KDDTest+.txt           # Test data
├── cicids2017/                 # CICIDS2017 dataset
│   └── *.csv                   # Multiple CSV files
├── kdd99/                      # Original KDD99
├── unsw_nb15/                  # UNSW-NB15 dataset
└── cyber_attacks/              # Additional attack data
```

**Label column**: `label` or `attack_cat` in CSV
- Normal traffic = 0
- Attack traffic = 1 (DoS, Probe, R2L, U2R, etc.)

---

## 4. FRAUD DETECTION (XGBoost)

**Purpose**: Detect fraudulent financial transactions

### Data (Mixed labels in CSV)
```
datasets/fraud_detection/
└── creditcard.csv              # ~285k transactions
```

**Label column**: `Class`
- Normal = 0
- Fraud = 1

---

## 5. TIMESERIES LSTM

**Purpose**: Detect temporal anomalies in network traffic patterns

### Data (Pre-generated numpy arrays)
```
datasets/timeseries/
├── normal_traffic_improved.npy   # Normal traffic sequences
└── attack_traffic_improved.npy   # Attack traffic sequences
```

**Note**: Currently uses synthetic data. Could be enhanced with:
- DARPA dataset (`datasets/darpa/`)
- CICIDS2017 temporal features

---

## 6. HOST BEHAVIOR DETECTOR (Optional)

**Purpose**: Detect malware based on memory/process behavior

### Data
```
datasets/cic_malmem/
├── Output1.csv
├── output2.csv
└── output3.csv

datasets/cic_malmem_full/
└── MalMem2022.csv              # Full CIC-MalMem-2022 dataset
```

**Label column**: `Category` or similar
- Benign = 0
- Malware = 1 (Spyware, Ransomware, Trojan, etc.)

---

## 7. HOLDOUT/TEST DATA (DO NOT TRAIN)

```
datasets/holdout_test/
├── holdout_test.jsonl          # Holdout test set
└── metadata.json               # Test set metadata

datasets/fp_test_500k.jsonl     # FP testing only
```

---

## Data Size Summary

| Dataset | Size | Samples |
|---------|------|---------|
| benign_60m/ | 4.5 GB | ~60M |
| security_payloads/ | ~1.8 GB | ~5M+ |
| url_analysis/ | ~50 MB | ~1.2M |
| network_intrusion/ | ~200 MB | ~500k |
| fraud_detection/ | 150 MB | 285k |
| timeseries/ | 55 MB | 30k |
| cic_malmem/ | 20 MB | 60k |

---

## Training Recommendations

### Payload CNN
- Use streaming dataset (data too large for memory)
- Balance: 50% malicious, 50% benign per batch
- Include adversarial benign samples to reduce FP
- Exclude passwords/wordlists (not injection attacks)

### URL CNN  
- Can fit in memory (~30M URLs)
- Balance malicious/benign
- Include hard synthetic samples

### Network Intrusion
- Sample from each dataset (50k per file)
- Stratified sampling to balance attack types
- Use unified feature set across datasets

### Fraud Detection
- Handle class imbalance (0.17% fraud)
- Use SMOTE or class weights
- Full dataset fits in memory

### Timeseries LSTM
- Generate more synthetic data if needed
- Consider using real DARPA/CICIDS temporal data
- Sequence length: 60 timesteps

---

## File Format Reference

| Format | Description |
|--------|-------------|
| `.jsonl` | JSON Lines - one JSON object per line with `text` and `label` |
| `.txt` | Plain text - one sample per line |
| `.csv` | CSV with headers |
| `.npy` | NumPy array |
