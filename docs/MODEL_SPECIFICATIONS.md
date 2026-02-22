# Model Specifications Summary

## PyTorch Models

### PayloadCNN
**Purpose**: Character-level CNN for malicious payload detection (SQL injection, XSS, command injection)

**Architecture**:
- Embedding layer: vocab_size=256, embed_dim=128
- Multi-scale convolutions:
  - Conv1d: 128 → 128 filters, kernel_size=3
  - Conv1d: 128 → 256 filters, kernel_size=5
  - Conv1d: 256 → 256 filters, kernel_size=7
- Adaptive max pooling
- FC layers: 256 → 128 → 1

**Parameters**: ~744,385 (approx)

**Input**: (batch, seq_len=500) of character indices
**Output**: Logits (binary classification)

---

### URLCNN
**Purpose**: Character-level CNN for malicious URL detection

**Architecture**:
- Embedding layer: vocab_size=128, embed_dim=64
- 3 parallel convolutions (kernel sizes: 3, 5, 7), each producing 64 filters
- Adaptive max pooling per branch
- Concatenation of pooled outputs
- FC layers: 192 → 64 → 1

**Parameters**: ~41,089 (approx)

**Input**: (batch, seq_len=200) of character indices
**Output**: Logits (binary classification)

---

### TimeSeriesLSTM
**Purpose**: Bidirectional LSTM for network traffic anomaly detection

**Architecture**:
- Bidirectional LSTM: input_dim=8, hidden_dim=64, num_layers=2, dropout=0.4
- FC layers: 128 → 32 → 1 (128 = 64*2 for bidirectional)

**Parameters**: ~51,201 (approx)

**Input**: (batch, seq_len=60, input_dim=8) time-series features
**Output**: Logits (binary classification)

---

### MetaClassifier
**Purpose**: Neural meta-classifier combining outputs from all detection models

**Architecture**:
- FC layers: 5 → 32 → 16 → 1
- BatchNorm1d after first FC
- Dropout: 0.2

**Parameters**: ~1,089 (approx)

**Input**: (batch, num_models=5) probability scores from each model
**Output**: Logits (binary classification)

---

## sklearn Models

### Network Intrusion RandomForest
**Purpose**: Detect network intrusion attacks using KDD Cup 99 features

**Configuration**:
- Algorithm: RandomForestClassifier
- n_estimators: 100
- max_depth: 20
- n_jobs: -1 (parallel)
- random_state: 42

**Features**: 37 network traffic features
- Duration, bytes, flags, login attempts, compromised indicators, etc.

**Data**:
- Malicious: Synthetic 500k samples
- Benign: MAWI live network traces

**Preprocessing**: StandardScaler normalization

---

### Host Behavior RandomForest
**Purpose**: Detect malware through host behavior analysis

**Configuration**:
- Algorithm: RandomForestClassifier
- n_estimators: 100
- max_depth: 20
- n_jobs: -1 (parallel)
- random_state: 42

**Features**: Dynamic (extracted from dataset, excludes 'label' and 'category')

**Data**:
- Malicious: Synthetic 500k samples
- Benign: Live benign host behavior traces

**Preprocessing**: StandardScaler normalization

---

### Fraud Detection XGBoost
**Purpose**: Detect fraudulent transactions

**Configuration**:
- Algorithm: XGBClassifier
- n_estimators: 100
- max_depth: 6
- learning_rate: 0.1
- n_jobs: -1 (parallel)
- random_state: 42

**Features**: Dynamic (extracted from dataset, excludes 'Class' label)

**Data**:
- Malicious: Synthetic 500k samples
- Benign: Live benign fraud dataset

**Preprocessing**: StandardScaler normalization

---

## Summary Table

| Model | Type | Parameters | Input Shape | Output |
|-------|------|-----------|-------------|--------|
| PayloadCNN | PyTorch CNN | ~744k | (batch, 500) | Logits |
| URLCNN | PyTorch CNN | ~41k | (batch, 200) | Logits |
| TimeSeriesLSTM | PyTorch LSTM | ~51k | (batch, 60, 8) | Logits |
| MetaClassifier | PyTorch MLP | ~1k | (batch, 5) | Logits |
| Network Intrusion RF | sklearn RF | 100 trees, depth 20 | 37 features | Probability |
| Host Behavior RF | sklearn RF | 100 trees, depth 20 | Dynamic | Probability |
| Fraud Detection XGB | XGBoost | 100 trees, depth 6 | Dynamic | Probability |

**Total PyTorch Parameters**: ~837,764

---

## Training Details

### PyTorch Models
- Loss: Binary Cross-Entropy (BCEWithLogitsLoss)
- Optimizer: Adam
- Device: CUDA (GPU)
- Batch size: Configurable (typically 32-64)

### sklearn/XGBoost Models
- Train/Test split: 80/20
- Preprocessing: StandardScaler
- Evaluation: Accuracy, Classification Report
- Serialization: joblib (.pkl format)

---

## Model Artifacts Location

```
models/
├── payload_cnn.pt              # PyTorch checkpoint
├── url_cnn.pt                  # PyTorch checkpoint
├── timeseries_lstm.pt          # PyTorch checkpoint
├── meta_classifier.pt          # PyTorch checkpoint
├── network_intrusion_model.pkl # sklearn model
├── network_scaler.pkl          # sklearn scaler
├── host_behavior_model.pkl     # sklearn model
├── host_behavior_scaler.pkl    # sklearn scaler
├── fraud_detection_model.pkl   # XGBoost model
└── fraud_scaler.pkl            # sklearn scaler
```

---

## Notes

- All PyTorch models output **logits** (no sigmoid/softmax applied) for use with BCEWithLogitsLoss
- sklearn/XGBoost models output **probabilities** directly
- MetaClassifier expects normalized probability scores (0-1 range) from individual models
- sklearn models use StandardScaler for feature normalization; scaler must be applied at inference time
- Inference uses per-model thresholds from `config/model_thresholds.json` (defaults to 0.5)
- All models use random_state=42 for reproducibility
