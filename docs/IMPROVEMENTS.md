# Model Improvements - Detailed Implementation Plan

## Current State Summary

| Model | Accuracy | Issue | Root Cause |
|-------|----------|-------|------------|
| Payload CNN | 86.99% | High false positives (60% on "hello world") | Poor benign training data |
| URL CNN | 100% synthetic | Won't generalize to real URLs | Synthetic patterns too obvious |
| Time-Series LSTM | 100% synthetic | Overfitted, predicts 100% for everything | Attack patterns too extreme |
| sklearn models | Not loading | Error 118 | Git LFS files not pulled |

---

## Phase A: Data Quality Improvements (Priority: CRITICAL)

### A1: Curate Benign Payload Dataset
**Goal:** Reduce false positive rate from ~40% to <5%

**Problem Analysis:**
- Current benign data from wordlists contains passwords, subdomains, technical terms
- Model hasn't seen realistic user inputs (emails, names, sentences)
- "john.doe@example.com" triggers 100% because @ and dots look like injection

**Tasks:**
- [ ] Create `datasets/curated_benign/` directory structure
- [ ] Generate benign samples by category:

| Category | Count | Examples |
|----------|-------|----------|
| Natural sentences | 10,000 | "The weather is nice today", "Please confirm your order" |
| Names | 5,000 | "John Smith", "María García", "李明" |
| Email addresses | 5,000 | "user@company.com", "john.doe123@gmail.com" |
| Phone numbers | 5,000 | "+1-555-123-4567", "(555) 123-4567" |
| Addresses | 5,000 | "123 Main St, New York, NY 10001" |
| Dates | 5,000 | "2024-01-15", "January 15, 2024", "15/01/2024" |
| Product names | 5,000 | "iPhone 15 Pro Max", "Nike Air Jordan" |
| Search queries | 5,000 | "best restaurants near me", "python tutorial" |
| Form values | 5,000 | "username123", "MyP@ssw0rd!", "Remember me" |
| JSON/XML | 5,000 | `{"status": "ok", "count": 42}` |

- [ ] Create `scripts/generate_benign_data.py`
- [ ] Add data validation to ensure no malicious patterns leak in
- [ ] Retrain Payload CNN with balanced dataset
- [ ] Validate false positive rate on held-out benign samples

**Data Sources:**
- Sentences: Wikipedia, news articles, Gutenberg books
- Names: US Census data, international name databases
- Emails: Generated with realistic patterns
- Addresses: OpenAddresses dataset
- Search queries: AOL search logs (anonymized), Google trends

**Expected Outcome:** 95%+ accuracy, <5% false positive rate

**Estimated Time:** 3-4 days

---

### A2: Download Real URL Datasets
**Goal:** Train URL CNN on real-world phishing/malware URLs

**Problem Analysis:**
- Synthetic URLs have obvious patterns (.tk TLD, IP addresses)
- Real phishing uses legitimate TLDs, subtle typos, lookalike domains
- Model will fail on sophisticated attacks

**Tasks:**
- [ ] Download URLhaus (malware URLs, no auth needed):
  ```powershell
  curl -o datasets/url_analysis/urlhaus.csv https://urlhaus.abuse.ch/downloads/csv_recent/
  ```
- [ ] Download Tranco Top 1M (benign domains):
  ```powershell
  curl -o datasets/url_analysis/tranco_top1m.csv https://tranco-list.eu/download/X4J9Q/1000000
  ```
- [ ] Optional - PhishTank (requires free API key):
  ```powershell
  # Register at phishtank.org, then:
  curl -o datasets/url_analysis/phishtank.json "http://data.phishtank.com/data/YOUR_API_KEY/online-valid.json"
  ```
- [ ] Create `scripts/prepare_url_data.py` to parse and clean
- [ ] Balance classes (undersample benign if needed)
- [ ] Create train/val/test splits (80/10/10)
- [ ] Retrain URL CNN on real data

**Dataset Sizes:**
| Source | Type | Size | Format |
|--------|------|------|--------|
| URLhaus | Malware | ~100K URLs | CSV |
| Tranco | Benign | 1M domains | CSV |
| PhishTank | Phishing | ~50K URLs | JSON |

**Expected Outcome:** 90%+ accuracy on real-world URLs

**Estimated Time:** 2-3 days

---

### A3: Improve Synthetic URL Generation
**Goal:** Create harder synthetic URLs for better generalization

**Current Problems:**
```python
# Current (too obvious):
"http://paypal123.tk/login"        # Suspicious TLD
"http://192.168.1.1/malware.exe"   # IP address
"http://free-iphone.ml/claim"      # Suspicious keywords

# Real attacks (harder to detect):
"https://paypa1.com/signin"        # Typo (l→1)
"https://secure-paypal.com/verify" # Legitimate-looking
"https://paypal.com.evil.co/login" # Subdomain trick
```

**Tasks:**
- [ ] Update `generate_malicious_urls()` in `train_url.py`:
  - Use legitimate TLDs (.com, .org, .net, .co)
  - Single character typos (paypa1, arnazon, micros0ft)
  - Subdomain abuse (paypal.secure.evil.com)
  - Homograph-like patterns (rn→m, l→1, O→0)
  - Mix HTTP and HTTPS
  - Realistic paths (/account/verify, /signin, /secure)
  
- [ ] Update `generate_benign_urls()`:
  - More diverse legitimate domains (not just top 15)
  - Realistic query parameters
  - Various path structures
  - Include CDN URLs, API endpoints

- [ ] Add adversarial examples:
  - Benign-looking malicious URLs
  - Malicious-looking benign URLs (security blogs, etc.)

**Expected Outcome:** Model trained on harder data generalizes better

**Estimated Time:** 1-2 days

---

### A4: Improve Time-Series Synthetic Data
**Goal:** Create subtle attack patterns that don't cause 100% overfitting

**Current Problems:**
```python
# Current (too obvious):
packets[spike_start:] += 500-2000  # 5-20x spike
unique_ports[scan_start:] += 50-200  # Massive increase
error_rate[bf_start:] = 0.3-0.7  # 30-70% errors

# Real attacks (subtle):
packets[spike_start:] *= 1.5-2.0  # 50-100% increase
unique_ports[scan_start:] += 10-30  # Gradual increase
error_rate[bf_start:] = 0.05-0.15  # 5-15% errors
```

**Tasks:**
- [ ] Update `generate_attack_traffic()` in `train_timeseries.py`:
  - Reduce spike magnitudes (2x instead of 10x)
  - Add gradual ramp-up (over 10-20 timesteps)
  - Create intermittent patterns (attack, pause, attack)
  - Add "low and slow" variants
  - Mix attack signals with normal noise

- [ ] Add attack subtypes:
  - Slow DDoS (gradual increase over hours)
  - Stealthy port scan (1 port per minute)
  - Credential stuffing (low error rate, distributed)
  - Data exfiltration (small but consistent outbound)

- [ ] Add harder negatives:
  - Normal traffic spikes (legitimate high load)
  - Maintenance windows (unusual but benign)
  - Backup operations (high data transfer)

**Expected Outcome:** 85%+ accuracy with realistic detection

**Estimated Time:** 2-3 days

---

## Phase B: Model Architecture Improvements (Priority: HIGH)

### B1: Payload CNN Enhancements
**Goal:** Better feature extraction for edge cases

**Tasks:**
- [ ] Add self-attention layer after convolutions:
  ```python
  self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4)
  ```
- [ ] Implement focal loss for hard examples:
  ```python
  class FocalLoss(nn.Module):
      def __init__(self, alpha=0.25, gamma=2):
          # Focuses on hard-to-classify examples
  ```
- [ ] Add character bigram features alongside unigrams
- [ ] Experiment with larger embedding dimension (128→256)
- [ ] Add residual connections between conv layers

**Expected Outcome:** +2-3% accuracy improvement

**Estimated Time:** 3-4 days

---

### B2: URL Hybrid Scoring
**Goal:** Combine CNN with engineered features

**Tasks:**
- [ ] Create `url_feature_extractor.py`:
  ```python
  def extract_url_features(url):
      return {
          'length': len(url),
          'num_dots': url.count('.'),
          'num_hyphens': url.count('-'),
          'num_digits': sum(c.isdigit() for c in url),
          'has_ip': bool(re.match(r'\d+\.\d+\.\d+\.\d+', url)),
          'suspicious_tld': tld in ['.tk', '.ml', '.ga', '.xyz'],
          'entropy': calculate_entropy(domain),
          'subdomain_depth': url.count('.') - 1,
          'path_depth': url.count('/') - 2,
          'has_https': url.startswith('https'),
          # ... more features
      }
  ```
- [ ] Create hybrid model combining CNN + features:
  ```python
  class HybridURLModel(nn.Module):
      def __init__(self):
          self.cnn = URLCNN()
          self.feature_fc = nn.Linear(15, 32)
          self.combined_fc = nn.Linear(64 + 32, 1)
  ```
- [ ] Train hybrid model
- [ ] Compare with CNN-only performance

**Expected Outcome:** +3-5% accuracy on edge cases

**Estimated Time:** 3-4 days

---

### B3: Time-Series Architecture Improvements
**Goal:** Better temporal pattern detection

**Tasks:**
- [ ] Add attention mechanism:
  ```python
  class AttentionLSTM(nn.Module):
      def __init__(self):
          self.lstm = nn.LSTM(...)
          self.attention = nn.Linear(hidden_dim, 1)
      
      def forward(self, x):
          lstm_out, _ = self.lstm(x)
          attn_weights = F.softmax(self.attention(lstm_out), dim=1)
          context = (lstm_out * attn_weights).sum(dim=1)
          return self.fc(context)
  ```
- [ ] Implement reconstruction-based anomaly detection:
  - Train autoencoder on normal traffic only
  - Anomaly score = reconstruction error
- [ ] Add multi-scale features (1min, 5min, 15min windows)
- [ ] Consider Temporal Convolutional Network (TCN) as alternative

**Expected Outcome:** Better detection of subtle anomalies

**Estimated Time:** 4-5 days

---

## Phase C: sklearn Model Recovery (Priority: HIGH)

### C1: Fix sklearn Model Loading
**Goal:** Get sklearn models working in HybridPredictor

**Problem:** Error 118 indicates files are Git LFS pointers, not actual models

**Tasks:**
- [ ] Check Git LFS status:
  ```powershell
  git lfs ls-files
  git lfs status
  ```
- [ ] Pull LFS files:
  ```powershell
  git lfs pull
  ```
- [ ] If LFS not available, retrain models:
  ```powershell
  python src/train_all_models.py
  ```
- [ ] Verify models load:
  ```python
  import joblib
  model = joblib.load('models/network_intrusion_model.pkl')
  print(type(model))
  ```

**Expected Outcome:** All sklearn models load successfully

**Estimated Time:** 1-2 hours

---

## Phase D: Integration Enhancements (Priority: MEDIUM)

### D1: Batch Processing Optimization
**Tasks:**
- [ ] Add batch inference to HybridPredictor
- [ ] Implement GPU memory-efficient batching
- [ ] Add async processing option for high throughput
- [ ] Benchmark: target 1000+ samples/second

### D2: Explainability
**Tasks:**
- [ ] Integrate SHAP for feature importance
- [ ] Add attention visualization for CNNs
- [ ] Create prediction explanation output:
  ```python
  {
      'prediction': 'MALICIOUS',
      'confidence': 0.95,
      'reasons': [
          'SQL keyword detected: SELECT',
          'Suspicious pattern: OR 1=1',
          'URL contains IP address'
      ]
  }
  ```

### D3: Monitoring & Logging
**Tasks:**
- [ ] Add inference latency tracking
- [ ] Log prediction distributions
- [ ] Implement drift detection
- [ ] Create simple monitoring dashboard

**Estimated Time:** 1 week total

---

## Phase E: Evaluation & Testing (Priority: MEDIUM)

### E1: Comprehensive Metrics
**Tasks:**
- [ ] Implement evaluation script with:
  - Precision, Recall, F1 per class
  - Confusion matrix visualization
  - ROC curves and AUC
  - Per-attack-type accuracy
- [ ] Create held-out test sets (never used in training)
- [ ] Establish baseline metrics for comparison

### E2: Real-World Validation
**Tasks:**
- [ ] Collect real attack samples from:
  - CTF challenges
  - Bug bounty reports (sanitized)
  - Security research papers
- [ ] Test on production-like traffic
- [ ] Measure false positive rate on real benign traffic
- [ ] Document failure cases for future improvement

**Estimated Time:** 1 week total

---

## Implementation Timeline

### Week 1 (Immediate Priority)
| Day | Task | Expected Outcome |
|-----|------|------------------|
| 1 | C1: Fix sklearn models | All models loading |
| 2-3 | A2: Download real URL data | URLhaus + Tranco ready |
| 4-5 | A1: Generate benign payloads | 50K curated samples |
| 6-7 | Retrain URL CNN + Payload CNN | Improved accuracy |

### Week 2
| Day | Task | Expected Outcome |
|-----|------|------------------|
| 1-2 | A3: Improve synthetic URLs | Harder training data |
| 3-4 | A4: Improve time-series data | Subtle attack patterns |
| 5-6 | Retrain all models | Better generalization |
| 7 | E1: Evaluation metrics | Baseline established |

### Week 3-4
| Day | Task | Expected Outcome |
|-----|------|------------------|
| 1-4 | B1: Payload CNN attention | +2-3% accuracy |
| 5-8 | B2: URL hybrid model | +3-5% accuracy |
| 9-10 | B3: Time-series attention | Better anomaly detection |

### Week 5-6
| Day | Task | Expected Outcome |
|-----|------|------------------|
| 1-3 | D1: Batch processing | 1000+ samples/sec |
| 4-6 | D2: Explainability | Prediction reasons |
| 7-10 | E2: Real-world testing | Production validation |

---

## Success Metrics

| Metric | Current | Target | Priority |
|--------|---------|--------|----------|
| Payload CNN Accuracy | 86.99% | 95%+ | HIGH |
| Payload CNN FP Rate | ~40% | <5% | CRITICAL |
| URL CNN (real data) | N/A | 90%+ | HIGH |
| Time-Series (subtle) | N/A | 85%+ | MEDIUM |
| Ensemble Accuracy | N/A | 95%+ | HIGH |
| Inference Latency | ~50ms | <100ms | LOW |
| Throughput | N/A | 1000/sec | LOW |

---

## Quick Start Commands

```powershell
# Phase C1: Fix sklearn models
git lfs pull
# OR
python src/train_all_models.py

# Phase A2: Download URL data
curl -o datasets/url_analysis/urlhaus.csv https://urlhaus.abuse.ch/downloads/csv_recent/
curl -o datasets/url_analysis/tranco.csv "https://tranco-list.eu/download/X4J9Q/1000000"

# Retrain models after data improvements
python src/training/train_payload.py
python src/training/train_url.py
python src/training/train_timeseries.py
python src/training/train_meta.py

# Test improvements
python src/test_hybrid.py
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Real URL data unavailable | Use improved synthetic generation |
| Benign data contains attacks | Add validation script to filter |
| Model accuracy decreases | Keep baseline models, A/B test |
| Training takes too long | Use smaller datasets for iteration |
| GPU memory issues | Reduce batch size, use gradient checkpointing |

---

*Last Updated: 2025-12-19*
*Next Action: Start with Phase C1 (fix sklearn models) and A2 (download URL data)*
