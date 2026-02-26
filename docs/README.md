# AI Hacking Detection ML System

A production-grade machine learning system for real-time cybersecurity threat detection using ensemble models, PyTorch neural networks, and multi-agent architecture with web dashboard and REST API.

## 🎯 Overview

This system implements a hybrid ML/DL approach for detecting cyber attacks including network intrusions, malicious URLs, payload injections, fraud, host behavior anomalies, and timeseries attacks using both traditional ML (scikit-learn, XGBoost, LightGBM) and deep learning (PyTorch CNN/LSTM) models.

**Key Features:**
- 7 specialized detection models (4 PyTorch + 3 sklearn)
- Real-time REST API with FastAPI
- Interactive Next.js dashboard with dark mode
- Comprehensive stress testing framework (v1.4)
- Transfer learning with progressive unfreezing
- Thermal guardian for GPU safety
- Hybrid meta-learner ensemble

## 🎯 Target Performance Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Accuracy | 98.9%+ | Overall classification accuracy |
| Recall | 98%+ | Attack detection rate |
| FP Rate | 2-3% | False positive rate |
| Explainability | Full | Detailed indicators + analyst checklists |

## 📊 Current Dataset Status

- **Total Dataset Size**: ~94.8GB (100M+ samples across all categories)
- **Live Benign Data**: 35GB+ (Wikipedia, GitHub, StackOverflow, Reddit, Enron emails, MAWI network)
- **Synthetic Data**: 5M+ malicious URLs, 500k network/fraud/host samples
- **FP Test Dataset**: 500k diverse benign samples
- **Validation Samples**: 7,100
- **Data Categories**: 7 core detection types + curated benign data
- **Latest Addition**: V1.4 stress test framework with 60 scenarios per model

## 🏗️ Architecture

### Core Detection Models

1. **Network Intrusion Model** (sklearn RandomForest)
   - 35 features: duration, bytes, connection stats, error rates
   - Detects: DoS, Probe, R2L, U2R attacks
   - Model: `network_intrusion_model.pkl` (368KB)

2. **URL Analysis Model** (PyTorch CNN)
   - Character-level CNN (200 char max)
   - Detects: Phishing, typosquatting, DGA, malware URLs
   - Model: `url_cnn.pt` (344KB)

3. **Payload Classifier** (PyTorch CNN)
   - Character-level CNN (500 char max)
   - Detects: SQLi, XSS, CMDi, path traversal, SSTI, XXE, LDAP
   - Model: `payload_cnn.pt` (2.9MB)

4. **Fraud Detection Model** (sklearn XGBoost)
   - 30 features: time, PCA components, amount
   - Detects: Card-not-present, account takeover, synthetic fraud
   - Model: `fraud_detection_model.pkl` (275KB)

5. **Host Behavior Model** (sklearn RandomForest)
   - 37 features: process lists, DLLs, handles, memory artifacts
   - Detects: Spyware, ransomware, trojans, rootkits, backdoors
   - Model: `host_behavior_model.pkl` (223KB)

6. **Timeseries Detector** (PyTorch LSTM)
   - 60 timesteps × 8 features
   - Detects: DDoS, port scans, exfiltration, C2, brute force
   - Model: `timeseries_lstm.pt` (564KB)

7. **Meta-Classifier** (PyTorch Ensemble)
   - Combines outputs from all 6 models
   - 5-input neural network for final verdict
   - Model: `meta_classifier.pt` (16KB)

### Model Pipeline
```
Input Data → Feature Engineering → Specialized Models → Calibration → Ensemble Voting → Explainability → Triage → Alert
```

## 🆕 Latest Features (v2.1 - February 2026)

### V1.4 Stress Test Framework
- **Comprehensive Testing**: 60 domain-specific scenarios per model
- **Hybrid Distribution**: 70% risk-weighted + 30% adaptive testing
- **Interactive Dashboard**: HTML dashboard with Chart.js visualizations
- **Performance Tracking**: P50/P95/P99 latency metrics per model
- **CLI Interface**: `python scripts/stress_test_v14.py --model <name>`

### Transfer Learning & Training
- **Progressive Unfreezing**: 3-stage gradual layer unfreezing (FC → Conv → Embed)
- **Thermal Guardian**: Background GPU temperature monitor (kills at ≥90°C)
- **Real URL Dataset**: URLhaus, Kaggle, Tranco CSVs for training
- **Validation Monitoring**: EarlyStopping with patience=3, separate validation files
- **Graceful Shutdown**: SIGTERM handler for clean checkpoint saves

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

### Web Dashboard & API
- **Next.js Dashboard**: Interactive dark mode UI with real-time scanning
- **FastAPI Backend**: RESTful API for model predictions
- **Multi-Model Support**: URL, Payload, Batch, and History views
- **Zustand State Management**: Client-side state with persistence

## 📊 Datasets (7 Core Categories)

### 1. Network Intrusion
- NSL-KDD, CICIDS2017, UNSW-NB15, KDD99
- Attack types: DoS, Probe, R2L, U2R
- 500k+ synthetic samples

### 2. URL Analysis
- Kaggle malicious URLs (194,798 samples)
- URLhaus dataset (25,454 samples)
- Tranco top-1m (999,999 legitimate domains)
- Synthetic benign/malicious hard samples (50k each)
- 5M+ malicious URLs for training

### 3. Security Payloads
- Wordlists: 1.8GB+ (passwords, usernames, fuzzing payloads)
- Injection attacks, command tutorials, HTML educational content
- Default credentials for 500+ router models

### 4. Curated Benign Data
- 11 categories: sentences, names, emails, phones, addresses, dates, usernames, products, search queries, comments, JSON
- Adversarial benign samples (code snippets, SQL benign, math expressions, etc.)
- 60M+ generated benign samples

### 5. Live Benign Data (35GB+)
- Wikipedia text (8GB+)
- GitHub code snippets (26GB+)
- StackOverflow posts (107MB)
- Reddit comments (7MB)
- Enron emails (380MB)
- MAWI network traffic (994MB)
- Common Crawl URLs (882MB)

### 6. Fraud Detection
- Credit card transaction data (150MB)
- 500k+ synthetic fraud samples

### 7. Host Behavior
- CIC-MalMem-2022 dataset
- 500k+ synthetic host behavior samples
- 5GB+ live benign host data

### 8. Timeseries
- 500k+ attack/normal traffic samples
- 60 timesteps × 8 features per sample

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
python src/train_network_intrusion.py
python src/train_fraud_detection.py
python src/train_host_behavior.py

# Train PyTorch models
python src/training/train_payload.py
python src/training/train_url.py
python src/training/train_timeseries.py
python src/training/train_meta.py

# RTX 3050 optimized training with transfer learning
python scripts/train_rtx3050.py

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
│   ├── metrics_tracker.py        # Accuracy/recall/FP tracking
│   ├── threshold_optimizer.py    # Threshold optimization
│   ├── confidence.py             # Probability calibration
│   ├── context_classifier.py     # Context-aware FP reduction
│   ├── indicators.py             # Human-readable indicators
│   ├── explainer.py              # Unified explanation engine
│   ├── checklist.py              # Analyst checklist generator
│   ├── triage.py                 # Fast triage system
│   ├── torch_models/             # PyTorch architectures
│   ├── training/                 # Training utilities
│   │   ├── train_payload.py     # Payload CNN training
│   │   ├── train_url.py         # URL CNN training
│   │   ├── train_timeseries.py  # LSTM training
│   │   ├── train_meta.py        # Meta-classifier training
│   │   ├── transfer_learning.py # Progressive unfreezing
│   │   └── checkpoint.py        # Checkpoint management
│   ├── data/                     # Data loaders
│   │   ├── url_dataset.py       # Real URL dataset loader
│   │   ├── streaming_dataset.py # Memory-efficient streaming
│   │   └── benign_generators.py # Benign data generation
│   ├── stress_test/              # Stress testing framework
│   │   ├── v14/                 # V1.4 implementation
│   │   │   ├── runner.py        # Test runner
│   │   │   ├── scenarios.py     # Scenario management
│   │   │   ├── models.py        # Model wrappers
│   │   │   ├── dashboard.py     # HTML dashboard generator
│   │   │   └── logger.py        # Logging utilities
│   │   ├── runner.py            # Legacy runner
│   │   ├── metrics.py           # Performance metrics
│   │   └── reporter.py          # Report generation
│   ├── api/                      # FastAPI backend
│   │   ├── server.py            # API server
│   │   ├── routes/              # API routes
│   │   └── schemas.py           # Pydantic schemas
│   ├── alerts/                   # Alert system
│   │   ├── dispatcher.py        # Alert dispatcher
│   │   └── channels/            # Alert channels
│   └── agents/                   # Agent implementations
│       └── host_behavior_detector.py
├── scripts/                       # Utility scripts
│   ├── generate_benign_data.py   # Benign data generation
│   ├── generate_adversarial_benign.py
│   ├── generate_500k_benign_test.py  # 500k FP test data
│   ├── establish_baseline.py     # Baseline metrics
│   ├── create_holdout_set.py     # Holdout test set
│   ├── validate_metrics.py       # Final validation
│   ├── download_url_datasets.py  # URL dataset download
│   ├── evaluate_models.py        # Model evaluation
│   ├── validate_realworld.py     # Real-world validation
│   ├── retrain_all.py            # Batch retraining
│   ├── train_rtx3050.py          # RTX 3050 optimized training
│   ├── thermal_guardian.py       # GPU temperature monitor
│   ├── collect_model_outputs.py  # Meta-learner data collection
│   ├── stress_test_v14.py        # V1.4 stress test CLI
│   └── collect_live_data/        # Live data collection scripts
├── configs/                       # Configuration files
│   ├── optimal_thresholds.json   # Per-model thresholds
│   ├── training_rtx3050.yaml     # RTX 3050 training config
│   ├── alert_thresholds.yaml     # Alert thresholds
│   ├── training_config.yaml      # General training config
│   └── scenarios_v14/            # V1.4 stress test scenarios
│       ├── payload.yaml
│       ├── url.yaml
│       ├── timeseries.yaml
│       ├── meta.yaml
│       ├── network.yaml
│       ├── host.yaml
│       └── fraud.yaml
├── dashboard/                     # Next.js dashboard
│   ├── src/
│   │   ├── app/                  # Next.js app router
│   │   │   ├── scanner/         # Scanner page
│   │   │   ├── batch/           # Batch processing page
│   │   │   ├── history/         # History page
│   │   │   └── models/          # Model info page
│   │   ├── components/          # React components
│   │   │   ├── dashboard/       # Dashboard components
│   │   │   ├── scanner/         # Scanner components
│   │   │   ├── layout/          # Layout components
│   │   │   └── ui/              # UI components
│   │   ├── stores/              # Zustand stores
│   │   ├── hooks/               # Custom hooks
│   │   ├── lib/                 # Utilities
│   │   └── types/               # TypeScript types
│   ├── package.json
│   └── tailwind.config.ts
├── models/                        # Trained models (.pkl, .pt, .pth)
│   ├── network_intrusion_model.pkl (368KB)
│   ├── url_cnn.pt (344KB)
│   ├── payload_cnn.pt (2.9MB)
│   ├── fraud_detection_model.pkl (275KB)
│   ├── host_behavior_model.pkl (223KB)
│   ├── timeseries_lstm.pt (564KB)
│   └── meta_classifier.pt (16KB)
├── datasets/                      # Training datasets (94.8GB)
│   ├── network_intrusion/        # NSL-KDD, CICIDS2017, etc.
│   ├── url_analysis/             # Malicious URLs, Tranco
│   ├── security_payloads/        # Wordlists, payloads
│   ├── curated_benign/           # Generated benign data
│   ├── live_benign/              # 35GB+ live benign data
│   ├── fp_test_500k.jsonl        # 500k FP test samples
│   ├── holdout_test/             # Holdout test set
│   ├── fraud_detection/          # Credit card data
│   ├── host_behavior/            # Host behavior data
│   └── timeseries/               # Temporal data
├── checkpoints/                   # Training checkpoints
│   ├── payload/
│   ├── url/
│   ├── timeseries/
│   └── meta/
├── evaluation/                    # Evaluation reports
│   ├── validation_report.json
│   ├── baseline_report.json
│   └── metrics_logs/
├── alerts/                        # Generated alerts
├── forensics/                     # Incident logs
└── tests/                         # Unit tests
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

Last Updated: February 21, 2026
