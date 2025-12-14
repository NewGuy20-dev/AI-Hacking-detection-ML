# AI Hacking Detection - Development Plan

## Hardware Constraints
- **CPU**: 4 cores
- **RAM**: 16 GB
- **Storage**: 32 GB
- **GPU**: None (Codespace)

---

## Phase 1: Data Preparation

### 1.1 Dataset Inventory
| Dataset | Type | Size | Features |
|---------|------|------|----------|
| NSL-KDD | Network intrusion | ~150K records | 41 features |
| KDD99 | Network intrusion | ~5M records | 41 features |
| CICIDS2017 | Intrusion detection | ~2.8M records | 78 features |
| UNSW-NB15 | Network traffic | ~2.5M records | 49 features |
| Phishing | URLs/websites | ~11K records | 30 features |
| Spam corpus | Email content | ~6K emails | Text |
| Malware URLs | Malicious URLs | Variable | URL strings |
| Alexa Top 1M | Benign domains | 1M domains | Domain strings |

### 1.2 Data Loading Strategy
```python
# Memory-safe loading
- Load one dataset at a time
- Sample large datasets (KDD99, CICIDS, UNSW) to 100K records
- Use dtype='float32' to halve memory usage
- Delete after processing: del df; gc.collect()
```

### 1.3 Feature Engineering

**Network Datasets (NSL-KDD, KDD99, CICIDS, UNSW)**:
- Standardize to 41 common features
- Encode categoricals (protocol_type, service, flag)
- Normalize numerical features (MinMaxScaler)

**URL/Domain Datasets**:
| Feature | Description |
|---------|-------------|
| url_length | Total URL length |
| domain_length | Domain name length |
| path_depth | Number of '/' in path |
| special_char_count | Count of @, -, _, etc. |
| digit_ratio | Digits / total chars |
| entropy | Shannon entropy of URL |
| has_ip | Contains IP address |
| suspicious_tld | .xyz, .top, .tk, etc. |

**Phishing/Spam**:
- TF-IDF vectorization (max 1000 features)
- Extract: link count, urgency words, sender reputation

### 1.4 Unified Schema
```
Final feature vector per detector:
- Network Detector: 41 features
- URL Detector: 15 features  
- Content Detector: 1000 features (TF-IDF)
```

---

## Phase 2: Model Architecture

### 2.1 Model Selection
| Model | Use Case | Reason |
|-------|----------|--------|
| RandomForest | Network traffic | Handles mixed features, parallelizes well |
| LightGBM | URL/domain analysis | Fast, low memory, good with imbalanced data |
| XGBoost | Phishing/spam | Strong with sparse TF-IDF features |

### 2.2 Multi-Detector System
```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
├─────────────────┬─────────────────┬─────────────────────────────┤
│ Network Traffic │ URLs/Domains    │ Email/Content               │
│ (flow data)     │ (strings)       │ (text)                      │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         ▼                 ▼                       ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│ Feature Extract │ │ Feature Extract │ │ Feature Extract         │
│ 41 features     │ │ 15 features     │ │ TF-IDF (1000 features)  │
└────────┬────────┘ └────────┬────────┘ └──────────────┬──────────┘
         │                   │                         │
         ▼                   ▼                         ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────┐
│ DETECTOR 1      │ │ DETECTOR 2      │ │ DETECTOR 3              │
│ RandomForest    │ │ LightGBM        │ │ XGBoost                 │
│ n_estimators=100│ │ num_leaves=31   │ │ max_depth=6             │
│ max_depth=20    │ │ n_estimators=100│ │ n_estimators=100        │
│ n_jobs=4        │ │ n_jobs=4        │ │ n_jobs=4                │
└────────┬────────┘ └────────┬────────┘ └──────────────┬──────────┘
         │                   │                         │
         │ prob_network      │ prob_url                │ prob_content
         │                   │                         │
         └───────────────────┼─────────────────────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   ENSEMBLE LAYER    │
                  │                     │
                  │ Weighted Average:   │
                  │ - Network: 0.5      │
                  │ - URL: 0.3          │
                  │ - Content: 0.2      │
                  │                     │
                  │ OR Meta-classifier  │
                  │ (Logistic Regress.) │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │   FINAL OUTPUT      │
                  │                     │
                  │ - is_attack: 0/1    │
                  │ - confidence: 0-1   │
                  │ - attack_type: str  │
                  └─────────────────────┘
```

### 2.3 Hyperparameters (CPU-Optimized)
```python
# RandomForest (Network)
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    n_jobs=4,
    random_state=42
)

# LightGBM (URL)
lgb.LGBMClassifier(
    n_estimators=100,
    num_leaves=31,
    max_depth=15,
    learning_rate=0.1,
    n_jobs=4
)

# XGBoost (Content)
xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    n_jobs=4,
    tree_method='hist'  # CPU optimized
)
```

---

## Phase 3: Training Pipeline

### 3.1 Directory Structure
```
AI-Hacking-detection-ML/
├── data/
│   ├── processed/
│   │   ├── network_train.csv
│   │   ├── network_test.csv
│   │   ├── url_train.csv
│   │   ├── url_test.csv
│   │   ├── content_train.csv
│   │   └── content_test.csv
├── models/
│   ├── network_detector.pkl
│   ├── url_detector.pkl
│   ├── content_detector.pkl
│   └── ensemble.pkl
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train_network.py
│   ├── train_url.py
│   ├── train_content.py
│   ├── ensemble.py
│   └── predict.py
└── notebooks/
    └── exploration.ipynb
```

### 3.2 Training Order
```
Step 1: Preprocess all datasets
        └── Output: processed/*.csv

Step 2: Train Network Detector
        └── Input: network_train.csv
        └── Output: network_detector.pkl

Step 3: Train URL Detector  
        └── Input: url_train.csv
        └── Output: url_detector.pkl

Step 4: Train Content Detector
        └── Input: content_train.csv
        └── Output: content_detector.pkl

Step 5: Train Ensemble
        └── Input: All detector predictions on validation set
        └── Output: ensemble.pkl
```

### 3.3 Memory Management
```python
# Pattern for each training script
import gc

def train_detector(data_path):
    # Load
    df = pd.read_csv(data_path, dtype='float32')
    
    # Train
    model = train(df)
    
    # Save
    joblib.dump(model, 'model.pkl')
    
    # Free memory
    del df
    gc.collect()
    
    return model
```

---

## Phase 4: Evaluation

### 4.1 Metrics
| Metric | Target | Priority |
|--------|--------|----------|
| Accuracy | >90% | Medium |
| Precision | >85% | High (reduce false positives) |
| Recall | >90% | High (catch attacks) |
| F1-Score | >87% | Primary metric |
| Inference time | <100ms | Required |

### 4.2 Evaluation Strategy
```
1. Per-detector evaluation
   - Confusion matrix
   - ROC-AUC curve
   - Precision-Recall curve

2. Ensemble evaluation
   - Compare vs individual detectors
   - Test on held-out data

3. Cross-validation
   - 5-fold stratified CV
   - Report mean ± std
```

### 4.3 Attack Type Classification
```
Binary: Normal vs Attack

Multi-class (if binary works):
- DoS (Denial of Service)
- Probe (Surveillance)
- R2L (Remote to Local)
- U2R (User to Root)
- Phishing
- Spam
- Malware URL
```

---

## Phase 5: Implementation Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1 | Data loader + preprocessing | `src/data_loader.py`, `src/feature_engineering.py` |
| 1 | Network detector | `src/train_network.py`, `models/network_detector.pkl` |
| 2 | URL detector | `src/train_url.py`, `models/url_detector.pkl` |
| 2 | Content detector | `src/train_content.py`, `models/content_detector.pkl` |
| 3 | Ensemble + evaluation | `src/ensemble.py`, `src/predict.py` |
| 3 | Testing + documentation | Updated README, final metrics |

---

## Phase 6: Future Enhancements (GPU)

When transferred to RTX 3050 laptop:

### 6.1 Model Upgrades
- Replace RandomForest with Neural Network
- Add LSTM for sequential traffic patterns
- Use 1D CNN for payload analysis
- Implement attention mechanisms

### 6.2 Scale Up
- Full datasets (no sampling)
- Deeper architectures
- Hyperparameter optimization (Optuna)
- Real-time streaming inference

### 6.3 TensorFlow Migration
```python
# Future GPU model
model = tf.keras.Sequential([
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## Quick Start Commands

```bash
# Install dependencies
pip install pandas scikit-learn xgboost lightgbm joblib

# Preprocess data
python src/data_loader.py

# Train all detectors
python src/train_network.py
python src/train_url.py
python src/train_content.py
python src/ensemble.py

# Run prediction
python src/predict.py --input sample.csv
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Out of memory | Sample datasets, use float32, gc.collect() |
| Slow training | Reduce n_estimators, limit max_depth |
| Imbalanced data | Use class_weight='balanced', SMOTE |
| Overfitting | Cross-validation, early stopping |
| Storage full | Compress models, delete intermediate files |
