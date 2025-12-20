# Dataset Download Guide

## Current Status

| Dataset | Status | Size | Location |
|---------|--------|------|----------|
| Security Payloads (FuzzDB, XSS, SQLi) | ✅ Have | 1.7 GB | `datasets/security_payloads/` |
| URLhaus Malicious URLs | ✅ Have | 4.8 MB | `datasets/url_analysis/urlhaus.csv` |
| Tranco Top 1M Domains | ✅ Have | 22 MB | `datasets/url_analysis/top-1m.csv` |
| Curated Benign Data | ✅ Have | 1.1 MB | `datasets/curated_benign/` |
| Synthetic URLs (Hard) | ✅ Have | 3.5 MB | `datasets/url_analysis/synthetic_*.txt` |
| Time-Series (Improved) | ✅ Have | 57 MB | `datasets/timeseries/*.npy` |
| NSL-KDD | ❌ LFS Pointer | ~25 MB | Need download |
| Kaggle Malicious URLs | ❌ Missing | ~65 MB | Optional |
| HuggingFace Phishing | ❌ Missing | ~50 MB | Optional |

---

## Datasets to Download

### 1. NSL-KDD (Required - ~25 MB)
Network intrusion detection dataset with 41 features.

**Source:** https://www.unb.ca/cic/datasets/nsl.html

**Download:**
```powershell
# From GitHub mirror
curl -o datasets/network_intrusion/nsl_kdd/KDDTrain+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt
curl -o datasets/network_intrusion/nsl_kdd/KDDTest+.txt https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt
```

---

### 2. Kaggle Malicious URLs (Optional - ~65 MB)
651K labeled URLs for malicious URL detection.

**Source:** https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset

**Download (GitHub mirror):**
```powershell
curl -o datasets/url_analysis/kaggle_malicious_urls.csv https://raw.githubusercontent.com/incertum/cyber-matrix-ai/master/Malicious-URL-Detection-Deep-Learning/data/url_data_mega_deep_learning.csv
```

---

### 3. HuggingFace Phishing Dataset (Optional - ~50 MB)
800K+ URLs with phishing labels.

**Source:** https://huggingface.co/datasets/ealvaradob/phishing-dataset

**Download (requires `datasets` library):**
```powershell
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('ealvaradob/phishing-dataset', 'url', trust_remote_code=True); ds['train'].to_csv('datasets/url_analysis/huggingface_phishing_urls.csv')"
```

---

### 4. PayloadsAllTheThings (Optional - ~15 MB)
Additional injection payloads for SQLi, XSS, SSTI, etc.

**Source:** https://github.com/swisskyrepo/PayloadsAllTheThings

**Download:**
```powershell
git clone --depth 1 https://github.com/swisskyrepo/PayloadsAllTheThings.git datasets/security_payloads/PayloadsAllTheThings
```

---

### 5. SecLists - Fuzzing Only (Optional - ~50 MB)
Selective download of fuzzing payloads.

**Source:** https://github.com/danielmiessler/SecLists

**Download (fuzzing folder only):**
```powershell
# Clone sparse checkout
git clone --depth 1 --filter=blob:none --sparse https://github.com/danielmiessler/SecLists.git datasets/security_payloads/SecLists
cd datasets/security_payloads/SecLists
git sparse-checkout set Fuzzing
```

---

## Quick Download Script

Run to download all missing datasets:
```powershell
python scripts/download_missing_datasets.py
```

---

## Dataset Details

### URL Datasets
| Source | Malicious | Benign | Total |
|--------|-----------|--------|-------|
| URLhaus | 25K | 0 | 25K |
| Tranco | 0 | 1M | 1M |
| Kaggle | ~450K | ~200K | 651K |
| HuggingFace | ~400K | ~400K | 800K |

### Payload Datasets
| Source | SQLi | XSS | Command Inj | Other |
|--------|------|-----|-------------|-------|
| FuzzDB | ✅ | ✅ | ✅ | ✅ |
| PayloadsAllTheThings | ✅ | ✅ | ✅ | ✅ |
| SecLists | ✅ | ✅ | ✅ | ✅ |

### Network Intrusion Datasets
| Dataset | Records | Features | Attack Types |
|---------|---------|----------|--------------|
| NSL-KDD | 150K | 41 | DoS, Probe, R2L, U2R |
| CIC-IDS2017 | Millions | 80+ | Modern attacks |

---

*Last Updated: 2025-12-20*
