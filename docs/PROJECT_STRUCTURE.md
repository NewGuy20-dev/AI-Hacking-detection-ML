# Project Structure

## Overview
Hybrid ML system combining sklearn/XGBoost/LightGBM (CPU) with PyTorch (GPU) for AI-powered hacking detection.

## Directory Structure

```
AI-Hacking-detection-ML/
├── models/                          # Trained models
│   ├── payload_cnn.pt               # PyTorch: Payload classifier
│   ├── url_cnn.pt                   # PyTorch: URL classifier
│   ├── timeseries_lstm.pt           # PyTorch: Time-series detector
│   ├── meta_classifier.pt           # PyTorch: Ensemble combiner
│   ├── network_intrusion_model.pkl  # sklearn: Network intrusion
│   ├── fraud_detection_model.pkl    # sklearn: Fraud detection
│   ├── url_analysis_model.pkl       # LightGBM: URL features
│   └── anomaly_detector.pkl         # sklearn: Anomaly detection
│
├── src/
│   ├── torch_models/                # PyTorch model definitions
│   │   ├── __init__.py
│   │   ├── payload_cnn.py           # CNN for payload detection
│   │   ├── url_cnn.py               # CNN for URL detection
│   │   ├── timeseries_lstm.py       # BiLSTM for time-series
│   │   ├── meta_classifier.py       # Neural ensemble
│   │   ├── datasets.py              # PyTorch Dataset classes
│   │   ├── utils.py                 # Training utilities
│   │   └── verify_setup.py          # GPU verification
│   │
│   ├── training/                    # Training scripts
│   │   ├── train_payload.py
│   │   ├── train_url.py
│   │   ├── train_timeseries.py
│   │   └── train_meta.py
│   │
│   ├── hybrid_predictor.py          # Main inference class
│   ├── test_hybrid.py               # Test suite
│   └── [existing sklearn modules]
│
├── datasets/                        # Training data
│   ├── security_payloads/           # Malicious payloads
│   ├── url_analysis/                # URL datasets
│   └── network_intrusion/           # Network traffic data
│
├── HYBRID_PYTORCH_PLAN.md           # Implementation plan
├── IMPROVEMENTS.md                  # Needed improvements
└── requirements.txt                 # Dependencies
```

## Model Architecture

### PyTorch Models (GPU)

| Model | Architecture | Parameters | Input |
|-------|-------------|------------|-------|
| PayloadCNN | Embed→Conv1D×3→FC | 738K | 500 chars |
| URLCNN | Embed→Conv1D×3→FC | 82K | 200 chars |
| TimeSeriesLSTM | BiLSTM→FC | 141K | 60×8 features |
| MetaClassifier | FC→BN→FC | 801 | 5 scores |

### sklearn Models (CPU)

| Model | Type | Use Case |
|-------|------|----------|
| network_intrusion | RandomForest | Network traffic |
| fraud_detection | XGBoost | Transaction fraud |
| url_analysis | LightGBM | URL features |
| anomaly_detector | IsolationForest | Zero-day detection |

## Usage

### Quick Start
```python
from src.hybrid_predictor import create_predictor

predictor = create_predictor('models')

result = predictor.predict({
    'payloads': ["SELECT * FROM users"],
    'urls': ["http://suspicious.tk/login"]
})

print(result['is_attack'])    # [1] or [0]
print(result['confidence'])   # [0.95]
print(result['scores'])       # Component scores
```

### Training
```bash
# Train all PyTorch models
python src/training/train_payload.py
python src/training/train_url.py
python src/training/train_timeseries.py
python src/training/train_meta.py
```

### Testing
```bash
python src/test_hybrid.py
```

## Requirements
- Python 3.10+
- PyTorch 2.1+ with CUDA
- scikit-learn, XGBoost, LightGBM
- NVIDIA GPU (RTX 3050 or better)

## Performance
- Inference: <50ms per sample
- GPU Memory: ~500MB
- Model Size: ~4MB total
