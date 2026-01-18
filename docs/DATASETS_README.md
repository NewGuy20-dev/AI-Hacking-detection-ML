# Datasets

Organized by detection category.

## Structure

```
datasets/
├── network_intrusion/     # 1.7 GB
│   ├── nsl_kdd/          # NSL-KDD (train/test)
│   ├── kdd99/            # KDD Cup 1999
│   ├── cicids2017/       # CICIDS 2017 (8 files)
│   ├── unsw_nb15/        # UNSW-NB15 (8 files)
│   └── cyber_attacks/    # Cyber Security Attacks
│
├── fraud_detection/       # 144 MB
│   └── creditcard.csv    # Credit Card Fraud
│
├── url_analysis/          # 50 MB
│   ├── malicious_urls/   # Malicious URLs + URLhaus
│   ├── phishing/         # Phishing dataset
│   └── domains/          # Alexa Top 1M
│
├── email_spam/            # 15 MB
│   └── spam_corpus       # Spam email corpus
│
└── security_payloads/     # 1.7 GB (8000+ files)
    ├── injection/        # XSS, SQLi, CRLF, Open Redirect
    ├── passwords/        # Default creds, password lists
    ├── wordlists/        # SecLists, dirsearch, subdomains
    ├── fuzzing/          # FuzzDB, IntruderPayloads, naughty strings
    └── misc/             # CVE tests, robots.txt, misc tools
```

## Total: ~3.6 GB across 8000+ files

## Models Trained

| Model | Dataset | Accuracy |
|-------|---------|----------|
| Network Intrusion | NSL-KDD + CICIDS2017 | 78.07% |
| Fraud Detection | Credit Card | 87.4% |
| URL Analysis | Malicious URLs | 88.94% |
| Payload Classifier | Security Payloads | 95.33% |
