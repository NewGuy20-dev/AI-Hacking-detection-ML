# AI Agents for Hacking Detection

## Ensemble Detection Agent

**Purpose**: Multi-model threat detection using weighted voting and meta-learning

**Capabilities**:
- Network intrusion detection (RandomForest)
- URL maliciousness classification (PyTorch CNN)
- Payload injection detection (PyTorch CNN)
- Fraud detection (XGBoost)
- Host behavior analysis (RandomForest)
- Timeseries anomaly detection (LSTM)
- Weighted ensemble voting
- Meta-classifier for final verdict (PyTorch)

**Input**: Network flows, URLs, text payloads, host metrics, timeseries data
**Output**: Attack probability + confidence score + explainability

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
- Character-level CNN analysis (200 char max)
- Phishing domain detection
- Typosquatting detection
- DGA (Domain Generation Algorithm) detection
- Malware URL classification
- URL shortener analysis
- Homograph attack detection

**Architecture**:
```
Input (char sequence) → Embedding → Conv1D layers → MaxPool → FC → Sigmoid
```

**Features**:
- Character-level encoding (0-127 ASCII)
- Multi-scale convolution features
- Real-time URL reputation scoring

**Model**: PyTorch CNN (344KB)
**Training Data**: URLhaus, Kaggle malicious URLs, Tranco top-1m, 5M+ synthetic URLs

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

## Host Behavior Agent

**Purpose**: Host-based intrusion detection

**Capabilities**:
- Process behavior analysis
- DLL injection detection
- Memory artifact analysis
- Malware classification (spyware, ransomware, trojans, rootkits, backdoors)

**Features**:
- 37 features from memory forensics
- Process lists, DLL lists, handles, memory artifacts
- Behavioral pattern recognition

**Model**: RandomForest (223KB)
**Training Data**: CIC-MalMem-2022, 500k+ synthetic samples

## Timeseries Anomaly Agent

**Purpose**: Temporal pattern analysis for network attacks

**Capabilities**:
- DDoS attack detection
- Port scan identification
- Data exfiltration detection
- C2 communication patterns
- Brute force attack recognition

**Architecture**:
```
Input (60 timesteps × 8 features) → LSTM layers → FC → Sigmoid
```

**Model**: PyTorch LSTM (564KB)
**Training Data**: 500k+ attack/normal traffic samples

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
1. **Data Collection** → loads datasets from 7 categories
2. **Feature Engineering** → extracts domain-specific features
3. **Model Training** → trains 7 specialized models (4 PyTorch + 3 sklearn)
4. **Ensemble Creation** → combines models with weights
5. **Validation** → cross-validation and metrics

**Training Scripts**:
- `train_all_models.py` - Unified training pipeline
- `train_rtx3050.py` - RTX 3050 optimized training with transfer learning
- `train_payload.py` - PyTorch CNN training for payloads
- `train_url.py` - PyTorch CNN training for URLs
- `train_timeseries.py` - LSTM training for temporal data
- `train_meta.py` - Meta-classifier training
- Individual sklearn model trainers

**Transfer Learning Features**:
- Progressive unfreezing (FC → Conv → Embed)
- Thermal guardian for GPU safety
- Validation monitoring with early stopping
- Graceful shutdown handlers

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
- **URL Model**: PyTorch CNN precision/recall optimization
- **Payload CNN**: Character-level detection accuracy 99.89%
- **Fraud Model**: XGBoost with class imbalance handling
- **Host Behavior**: RandomForest with memory forensics features
- **Timeseries LSTM**: Temporal pattern detection 75.38%
- **Ensemble**: Weighted voting with confidence thresholds

## Stress Testing Framework (V1.4)

**Purpose**: Comprehensive validation of all 7 models against domain-specific scenarios

**Features**:
- 60 scenarios per model (balanced static + dynamic generation)
- Hybrid distribution: 70% risk-weighted + 30% adaptive
- Performance tracking: P50/P95/P99 latency metrics
- Interactive HTML dashboard with Chart.js visualizations
- CLI interface: `python scripts/stress_test_v14.py --model <name>`

**Components**:
- `src/stress_test/v14/runner.py` - Test execution engine
- `src/stress_test/v14/scenarios.py` - Scenario management
- `src/stress_test/v14/models.py` - Model wrappers
- `src/stress_test/v14/dashboard.py` - HTML report generator
- `configs/scenarios_v14/` - YAML scenario definitions

**Tested Models**:
- PayloadCNN, URLCNN, TimeSeriesLSTM, MetaClassifier (PyTorch)
- FraudDetection, HostBehavior, NetworkIntrusion (sklearn)

---

## System Architecture & Development Guidelines

### **Project Overview**

Production-grade **cybersecurity threat detection system** using hybrid ML/DL ensemble models for real-time detection of network intrusions, malicious URLs, payload injections, fraud, host anomalies, and timeseries attacks.

**Target Metrics**: 98.9%+ accuracy, 98%+ recall, 2-3% FP rate

### **Technology Stack**

**Backend (Python)**
- **PyTorch 2.1+** - Deep learning models
- **scikit-learn** - Traditional ML (RandomForest, XGBoost)
- **FastAPI** - REST API server
- **Pandas/NumPy** - Data processing

**Frontend (TypeScript)**
- **Next.js 16** - React framework with App Router
- **React 19** - UI library
- **Tailwind CSS** - Styling
- **Zustand** - State management
- **Recharts** - Data visualization

### **Directory Structure**

```
AI-Hacking-detection-ML/
├── src/                          # Core source code
│   ├── torch_models/             # PyTorch architectures
│   ├── training/                 # Training utilities
│   ├── data/                     # Data loaders
│   ├── api/                      # FastAPI backend
│   ├── stress_test/v14/          # Stress testing framework
│   ├── ensemble.py               # Ensemble voting
│   ├── explainer.py              # Model explainability
│   └── triage.py                 # Alert triage
├── scripts/                      # Utility scripts
│   ├── train_rtx3050.py          # Main training script
│   ├── thermal_guardian.py       # GPU monitor
│   └── stress_test_v14.py        # Stress test CLI
├── dashboard/                    # Next.js frontend
│   └── src/app/                  # App router pages
├── models/                       # Trained models (.pkl, .pt)
├── datasets/                     # Training data (94.8GB)
├── checkpoints/                  # Training checkpoints
├── configs/                      # Configuration files
├── evaluation/                   # Evaluation reports
└── tests/                        # Unit tests (pytest)
```

### **Common Commands**

**Training**
```bash
# Train all models (RTX 3050 optimized)
python scripts/train_rtx3050.py

# Train specific models
python src/training/train_payload.py
python src/training/train_url.py
python src/training/train_timeseries.py
python src/training/train_meta.py

# Retrain all
python scripts/retrain_all.py
```

**API & Dashboard**
```bash
# Start API server
python src/api/server.py

# Dashboard (in dashboard/)
npm install
npm run dev      # Development
npm run build    # Production build
npm start        # Production server
```

**Testing**
```bash
# Run tests
pytest tests/

# Stress tests
python scripts/stress_test_v14.py --model payload

# Validation
python scripts/validate_realworld.py
```

### **Coding Conventions**

**Python**
- **Docstrings**: Triple-quoted for modules/classes/functions
- **Type hints**: Use `typing` module extensively
- **Classes**: PascalCase (`PayloadCNN`, `EnsembleDetector`)
- **Functions**: snake_case (`load_models`, `predict_proba`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_WEIGHTS`)
- **Private**: Leading underscore (`_handle_shutdown`)
- **Paths**: Use `pathlib.Path` over strings
- **Imports**: Group (stdlib → third-party → local)

**PyTorch**
- Models inherit from `nn.Module`
- Forward pass returns logits (no sigmoid in model)
- Device management: `.to(device)` pattern
- Checkpoints: `{model_state, optimizer_state, epoch, loss}`

**TypeScript/React**
- **Components**: PascalCase functional components
- **Hooks**: camelCase with `use` prefix
- **Props**: TypeScript interfaces
- **State**: Zustand stores for global state
- **Styling**: Tailwind utility classes

**Configuration**
- **YAML** for training configs
- **JSON** for thresholds/metrics
- **Environment variables** for secrets

### **Development Philosophy**

⚠️ **CRITICAL: Always Prioritize Proper Fixes Over Quick Patches**

When implementing changes or fixing issues:

1. **Root Cause Analysis First**
   - Identify the underlying problem, not just symptoms
   - Understand why the issue exists in the architecture
   - Consider long-term implications

2. **Proper Solutions Over Quick Fixes**
   - Refactor code properly rather than adding workarounds
   - Fix architectural issues at their source
   - Avoid technical debt accumulation
   - Don't patch over problems with temporary hacks

3. **Quality Standards**
   - Write clean, maintainable code
   - Add proper error handling
   - Include comprehensive tests
   - Document complex logic
   - Follow established patterns

4. **When to Refactor**
   - If you're adding a third conditional for the same issue → refactor
   - If you're copying code → create a reusable function
   - If you're working around a limitation → fix the limitation
   - If you're adding "TODO: fix this properly" → fix it now

5. **Technical Debt Management**
   - Address technical debt immediately when discovered
   - Don't defer proper fixes to "later"
   - Maintain code quality consistently
   - Refactor proactively, not reactively

**Examples:**
- ❌ Adding try-except to hide an error → ✅ Fix the error source
- ❌ Duplicating code with slight changes → ✅ Create parameterized function
- ❌ Adding special case handling → ✅ Redesign for generality
- ❌ Patching data format issues → ✅ Standardize data pipeline
- ❌ Working around API limitations → ✅ Improve API design

**Remember**: Quick patches create maintenance burden. Proper fixes create sustainable systems.

### **Special Features**

- **Transfer Learning**: Progressive unfreezing (FC → Conv → Embed)
- **Thermal Guardian**: GPU monitor kills training at ≥90°C
- **Graceful Shutdown**: SIGTERM handlers save checkpoints
- **Hybrid Meta-Learner**: 70% real + 30% synthetic outputs
- **Context-Aware Classification**: Reduces FPs via input context detection
- **Explainability**: Human-readable indicators + analyst checklists
- **Real-time API**: FastAPI with async lifecycle management
