# AI Agents for Hacking Detection

## Ensemble Detection Agent

**Purpose**: Multi-model threat detection using weighted voting

**Capabilities**:
- Network intrusion detection (RandomForest)
- URL maliciousness classification (LightGBM)
- Payload injection detection (PyTorch CNN)
- Fraud detection (XGBoost)
- Weighted ensemble voting
- Meta-classifier option

**Input**: Network flows, URLs, text payloads
**Output**: Attack probability + confidence score

## Network Intrusion Agent

**Purpose**: Real-time network traffic analysis

**Capabilities**:
- Binary classification (normal/attack)
- Multi-class attack type identification (DoS, Probe, R2L, U2R)
- Feature extraction from 41-dimensional network flows
- StandardScaler preprocessing

**Datasets**: NSL-KDD, CICIDS2017, UNSW-NB15, KDD99
**Model**: RandomForest with 100 estimators

## URL Analysis Agent

**Purpose**: Malicious URL detection and classification

**Capabilities**:
- URL feature extraction (length, special chars, TLD analysis)
- Phishing domain detection
- Malicious URL classification
- Domain reputation scoring

**Features**:
- URL length, path depth, special character counts
- Suspicious TLD detection (.xyz, .tk, .ml, etc.)
- Character distribution analysis

**Model**: LightGBM classifier

## Payload Classification Agent

**Purpose**: Injection attack detection using deep learning

**Capabilities**:
- Character-level CNN analysis
- SQL injection detection
- XSS payload identification
- Command injection recognition
- Multi-scale convolution features

**Architecture**:
```
Input (char sequence) → Embedding → Conv1D layers → MaxPool → FC → Sigmoid
```

**Model**: PyTorch CNN with 256 filters, 3-5-7 kernel sizes

## Fraud Detection Agent

**Purpose**: Financial transaction anomaly detection

**Capabilities**:
- Credit card fraud detection
- Transaction pattern analysis
- Real-time scoring
- Feature importance analysis

**Model**: XGBoost with 100 estimators, max_depth=6

## Alert Management Agent

**Purpose**: Structured alert generation and response

**Capabilities**:
- Severity classification (LOW, MEDIUM, HIGH, CRITICAL)
- Alert correlation and deduplication
- JSON-formatted alert output
- Timestamp and metadata tracking

**Alert Structure**:
- Unique alert ID
- Attack type classification
- Confidence score
- Source/destination IPs
- Severity level
- Timestamp

## Threat Intelligence Agent

**Purpose**: IOC database management and lookups

**Capabilities**:
- SQLite-based IOC storage
- Hash, IP, domain reputation checks
- Threat type classification
- Confidence scoring
- Source attribution

**Database Schema**:
- IOC type, value, threat_type
- Confidence score (0-100)
- Source and timestamp

## Forensic Logging Agent

**Purpose**: Automated evidence collection and incident logging

**Capabilities**:
- Compressed incident logs (.gz format)
- Evidence preservation
- Chain of custody tracking
- Automated log rotation

**Output**: Structured JSON logs in forensics/ directory

## Model Training Coordination

**Training Pipeline**:
1. **Data Collection** → loads datasets from 6 categories
2. **Feature Engineering** → extracts domain-specific features
3. **Model Training** → trains 4 specialized models
4. **Ensemble Creation** → combines models with weights
5. **Validation** → cross-validation and metrics

**Training Scripts**:
- `train_all_models.py` - Unified training pipeline
- `train_payload.py` - PyTorch CNN training
- Individual model trainers

## Agent Coordination

```
Input Data → Feature Extraction → Specialized Models → Ensemble Voting → Alert Generation
     ↓              ↓                    ↓                ↓              ↓
Data Loader → Feature Engineer → Model Router → Ensemble → Alert Manager
```

**Weights**: Network (0.5), URL (0.3), Content (0.2)

## Deployment Architecture

- **Real-time inference**: <100ms per prediction
- **Model updates**: Automated retraining pipeline
- **Threat database**: Continuous IOC updates
- **Monitoring**: 24/7 automated detection
- **Storage**: Models in .pkl/.pt format

## Performance Metrics

- **Network Model**: RandomForest accuracy >95%
- **URL Model**: LightGBM precision/recall optimization
- **Payload CNN**: Character-level detection accuracy
- **Fraud Model**: XGBoost with class imbalance handling
- **Ensemble**: Weighted voting with confidence thresholds
