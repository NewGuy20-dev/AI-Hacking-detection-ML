# AI Hacking Detection ML System

A comprehensive machine learning system for real-time cybersecurity threat detection using ensemble models, PyTorch neural networks, and multi-agent architecture.

## 🎯 Overview

This system implements a hybrid ML/DL approach for detecting cyber attacks including network intrusions, malicious URLs, payload injections, and fraud using both traditional ML (scikit-learn, XGBoost, LightGBM) and deep learning (PyTorch CNN/LSTM) models.

## 🎯 Target Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | 98.9%+ | Overall classification accuracy |
| Recall | 98%+ | Attack detection rate |
| FP Rate | 2-3% | False positive rate |
| Explainability | Full | Detailed indicators + analyst checklists |

## 📊 Current Dataset Status

- **Total Dataset Size**: 2.32 GB (95.2M+ samples)
- **FP Test Dataset**: 500k diverse benign samples
- **Validation Samples**: 7,100
- **Data Categories**: 6 core + curated benign data
- **Latest Addition**: 500k benign FP test dataset (Wikipedia, real-world text, edge cases, code, structured data)

## 🏗️ Architecture

### Core Detection Models

- **Network Intrusion Model**: RandomForest classifier for network traffic analysis
- **URL Analysis Model**: LightGBM + PyTorch CNN for malicious URL detection  
- **Payload Classifier**: PyTorch CNN for injection attack detection (character-level)
- **Fraud Detection Model**: XGBoost classifier for financial fraud
- **Timeseries Detector**: LSTM for temporal anomaly detection
- **Ensemble Detector**: Calibrated weighted voting with stacking meta-classifier

### Model Pipeline
```
Input Data → Feature Engineering → Specialized Models → Calibration → Ensemble Voting → Explainability → Triage → Alert
```

## 🆕 New Features (v2.0)

### High-Performance Detection
- **Threshold Optimization**: Grid search for optimal recall/FP tradeoff
- **Confidence Calibration**: Platt scaling and isotonic regression
- **Ensemble Stacking**: Meta-classifier for improved accuracy
- **Context-Aware Classification**: Reduces FPs by detecting input context (email, code, chat, etc.)

### Full Explainability
- **Indicators**: Human-readable detection indicators (SQL keywords, XSS patterns, etc.)
- **Explanations**: Verdict, confidence breakdown, attack type classification
- **Analyst Checklists**: Auto-generated verification steps per attack type

### Fast Triage
- **Priority Scoring**: P1-P5 priority levels with SLA hours
- **Quick Verdicts**: MALICIOUS, SUSPICIOUS, LIKELY_BENIGN, BENIGN
- **Auto-Actions**: Automatic blocking for high-confidence critical alerts
- **Batch Processing**: Efficient multi-alert triage

## 📊 Datasets (6 Core Categories)

### 1. Network Intrusion
- NSL-KDD, CICIDS2017, UNSW-NB15, KDD99
- Attack types: DoS, Probe, R2L, U2R

### 2. URL Analysis
- Kaggle malicious URLs (194,798 samples)
- URLhaus dataset (25,454 samples)
- Tranco top-1m (999,999 legitimate domains)
- Synthetic benign/malicious hard samples (50k each)

### 3. Security Payloads
- Wordlists: 1.8GB+ (passwords, usernames, fuzzing payloads)
- Injection attacks, command tutorials, HTML educational content
- Default credentials for 500+ router models

### 4. Curated Benign Data
- 11 categories: sentences, names, emails, phones, addresses, dates, usernames, products, search queries, comments, JSON
- Adversarial benign samples (code snippets, SQL benign, math expressions, etc.)

### 5. Email Spam
- Spam corpus with ham/spam classification

### 6. Fraud Detection
- Credit card transaction data

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup
```bash
# Generate benign data
python scripts/generate_benign_data.py

# Download URL datasets
python scripts/download_url_datasets.py

# Generate adversarial benign samples
python scripts/generate_adversarial_benign.py
```

### Training Models
```bash
# Train all models
python src/train_all_models.py

# Train specific models
python src/train_network.py
python src/train_url.py
python src/train_content.py

# Train PyTorch models
python src/training/train_payload.py
python src/training/train_url_cnn.py
python src/training/train_timeseries_lstm.py

# Retrain all models
python scripts/retrain_all.py
```

### Running Detection
```bash
# Real-time detection
python src/predict.py --input data.csv --type network

# URL analysis
python src/predict.py --input urls.txt --type url

# Payload detection
python src/predict.py --input payloads.txt --type content

# Ensemble prediction
python src/ensemble.py

# Batch prediction
python src/batch_predictor.py --input batch.csv

# Hybrid prediction (ML + DL)
python src/hybrid_predictor.py --input data.csv
```

### Validation & Evaluation
```bash
# Validate models
python src/validate.py

# Evaluate on real-world data
python scripts/validate_realworld.py

# Generate evaluation report
python scripts/evaluate_models.py
```

## 🔧 Key Features

### Detection Capabilities
- **Network Intrusion**: DoS, Probe, R2L, U2R attacks with 41-dimensional features
- **URL Analysis**: Malicious URL detection with character-level CNN + LightGBM
- **Payload Analysis**: CNN-based injection attack detection (SQL, XSS, command injection)
- **Fraud Detection**: Financial transaction analysis with XGBoost
- **Timeseries Anomaly**: LSTM-based temporal pattern detection
- **Anomaly Detection**: Unsupervised threat identification

### Advanced Features
- **Ensemble Voting**: Weighted combination of specialized models
- **PyTorch Deep Learning**: CNN for payloads/URLs, LSTM for timeseries
- **Alert Management**: Structured alert generation with severity scoring (LOW/MEDIUM/HIGH/CRITICAL)
- **Threat Intelligence**: IOC database integration with hash/IP/domain lookups
- **Forensic Logging**: Automated evidence collection with compression
- **Behavioral Profiling**: User/system behavior analysis
- **Explainability**: SHAP-based feature importance and model interpretability
- **Online Learning**: Continuous model updates with new data

## 📈 Performance Metrics

### Model Accuracy (Latest Training)
| Model | Validation Accuracy |
|-------|---------------------|
| Payload CNN | 99.89% |
| URL CNN | 97.47% |
| Time-Series LSTM | 75.38% |

### Validation Results: 92.9% (39/42 tests passed)
- **Payload Detection**: 89.3% (25/28)
- **URL Detection**: 100% (14/14)

### Known Limitations
The payload model may flag certain benign patterns as suspicious:

| Pattern | Behavior | Reason |
|---------|----------|--------|
| `<3` emoji (e.g., `<3 love this`) | False positive (~95%) | `<` character resembles HTML/XSS tag start |
| `SELECT * FROM menu` | Flagged as suspicious (~72%) | Ambiguous - could be SQL injection on restaurant sites |
| Emails with dots (e.g., `john.doe@example.com`) | Borderline (~52%) | Dot patterns can appear in injection payloads |

These are acceptable trade-offs for security - the model errs on the side of caution for ambiguous patterns.

### General Metrics
- **Real-time Inference**: <100ms per prediction
- **False Positive Rate**: <2%
- **Ensemble Precision/Recall**: Optimized for security use cases

## 🛠️ Model Components

### Core Models (in `/models/`)
- `network_intrusion_model.pkl` - RandomForest (239KB)
- `url_analysis_model.pkl` - LightGBM (87KB)
- `fraud_detection_model.pkl` - XGBoost (81KB)
- `payload_cnn.pt` - PyTorch CNN (2.97MB)
- `url_cnn.pt` - PyTorch CNN for URLs (346KB)
- `timeseries_lstm.pt` - LSTM for temporal data (576KB)
- `ensemble_voting.pkl` - Weighted ensemble
- `meta_classifier.pt` - Meta-learner for ensemble

### Feature Engineering
- Protocol analysis (TCP/UDP/ICMP)
- Connection statistics (duration, bytes, packets)
- Content-based features (entropy, special chars)
- Traffic pattern analysis
- URL structure analysis (length, depth, TLD, special chars)
- Character-level payload encoding
- Temporal features for timeseries

## 📁 Project Structure

```
├── src/                           # Source code
│   ├── train_*.py                # Model training scripts
│   ├── predict.py                # Prediction engine
│   ├── ensemble.py               # Ensemble methods (with calibration)
│   ├── hybrid_predictor.py       # ML + DL hybrid
│   ├── batch_predictor.py        # Batch processing
│   ├── alert_manager.py          # Alert generation (with explainability)
│   ├── threat_intel.py           # Threat intelligence
│   ├── monitoring.py             # Model monitoring
│   ├── metrics_tracker.py        # [NEW] Accuracy/recall/FP tracking
│   ├── threshold_optimizer.py    # [NEW] Threshold optimization
│   ├── confidence.py             # [NEW] Probability calibration
│   ├── context_classifier.py     # [NEW] Context-aware FP reduction
│   ├── indicators.py             # [NEW] Human-readable indicators
│   ├── explainer.py              # [NEW] Unified explanation engine
│   ├── checklist.py              # [NEW] Analyst checklist generator
│   ├── triage.py                 # [NEW] Fast triage system
│   ├── torch_models/             # PyTorch architectures
│   └── training/                 # Training utilities
├── scripts/                       # Utility scripts
│   ├── generate_benign_data.py   # Benign data generation
│   ├── generate_adversarial_benign.py
│   ├── generate_500k_benign_test.py  # [NEW] 500k FP test data
│   ├── establish_baseline.py     # [NEW] Baseline metrics
│   ├── create_holdout_set.py     # [NEW] Holdout test set
│   ├── validate_metrics.py       # [NEW] Final validation
│   ├── download_url_datasets.py  # URL dataset download
│   ├── evaluate_models.py        # Model evaluation
│   ├── validate_realworld.py     # Real-world validation
│   └── retrain_all.py            # Batch retraining
├── configs/                       # [NEW] Configuration files
│   └── optimal_thresholds.json   # Per-model thresholds
├── models/                        # Trained models (.pkl, .pt, .pth)
├── datasets/                      # Training datasets (2.32GB)
│   ├── network_intrusion/        # NSL-KDD, CICIDS2017, etc.
│   ├── url_analysis/             # Malicious URLs, Tranco
│   ├── security_payloads/        # Wordlists, payloads
│   ├── curated_benign/           # Generated benign data
│   ├── fp_test_500k.jsonl        # [NEW] 500k FP test samples
│   ├── holdout_test/             # [NEW] Holdout test set
│   ├── fraud_detection/          # Credit card data
│   ├── email_spam/               # Spam corpus
│   └── timeseries/               # Temporal data
├── evaluation/                    # Evaluation reports
├── alerts/                        # Generated alerts
└── forensics/                     # Incident logs
```

## 🔒 Security Features

### Threat Response
- Automated IP blocking recommendations
- Alert generation with confidence scores
- Risk assessment and severity classification
- Mitigation recommendations
- Incident logging with chain of custody

### Intelligence Integration
- Hash reputation checking (MD5, SHA1, SHA256)
- DNS analysis and domain reputation
- Geolocation tracking
- Threat actor profiling
- IOC database lookups

## 📊 Monitoring & Analytics

### Real-time Dashboards
- Threat detection rates
- Model performance metrics
- Attack pattern analysis
- Geographic threat mapping
- Model drift detection

### Reporting
- Executive summaries
- Technical incident reports
- Performance analytics
- Compliance reporting
- Feature importance analysis

## 🔄 Continuous Learning

- **Adaptive Models**: Self-updating based on new threats
- **Feedback Loop**: Human analyst input integration
- **A/B Testing**: Model performance comparison
- **Drift Detection**: Model degradation monitoring
- **Online Learning**: Incremental model updates

## 🚨 Alert Management

### Alert Types
- **CRITICAL**: Immediate response required (high confidence attacks)
- **HIGH**: Investigate within 1 hour
- **MEDIUM**: Review within 24 hours
- **LOW**: Routine monitoring

### Response Actions
- Automated blocking recommendations
- Analyst notification
- Evidence collection
- Remediation suggestions
- Forensic logging

## 📋 Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy
- joblib
- xgboost
- lightgbm
- torch>=2.1.0
- torchvision>=0.16.0
- matplotlib
- seaborn
- requests
- tqdm
- shap (for explainability)

## 🔗 Related Documentation

- [AGENTS.md](AGENTS.md) - Detailed agent architecture
- [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) - Development roadmap
- [ENHANCEMENT_PLAN.md](ENHANCEMENT_PLAN.md) - Future enhancements
- [DATASETS.md](DATASETS.md) - Dataset specifications
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed project layout

## 📞 Support

For issues and questions:
- Create GitHub issue
- Check documentation
- Review agent specifications
- See IMPROVEMENTS.md for recent updates

---

**Built for cybersecurity professionals by cybersecurity professionals** 🛡️

Last Updated: December 21, 2025
