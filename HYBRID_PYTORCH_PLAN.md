# Hybrid PyTorch + sklearn/XGBoost/LightGBM Implementation Plan

## Executive Summary

**Goal**: Enhance the AI Hacking Detection system with PyTorch deep learning while keeping sklearn/XGBoost/LightGBM for tabular data.

**Hardware**: NVIDIA RTX 3050 Laptop (4GB VRAM, 2048 CUDA cores, Compute 8.6)

**Timeline**: 7 weeks

**Expected Improvement**: Overall accuracy from ~95% to 97%+

---

## Why PyTorch Over TensorFlow?

| Aspect | PyTorch Advantage |
|--------|-------------------|
| Debugging | Eager execution, use Python debugger directly |
| Code style | More Pythonic, fits with existing sklearn code |
| Custom models | Cleaner forward() method, less boilerplate |
| Research | Most security ML papers implement in PyTorch |
| Memory control | More explicit CUDA memory management |
| Deployment | TorchScript + ONNX for flexibility |
| PyTorch 2.0 | torch.compile() for automatic optimization |

---

## Part 1: Current State (Keep These)

### Existing sklearn/XGBoost/LightGBM Models
| Model | Framework | Use Case | Status |
|-------|-----------|----------|--------|
| RandomForest | sklearn | Network intrusion (41 features) | KEEP |
| XGBoost | xgboost | Fraud detection | KEEP |
| LightGBM | lightgbm | URL feature analysis | KEEP |
| IsolationForest | sklearn | Anomaly detection | KEEP |
| SGDClassifier | sklearn | Online learning | KEEP |

### Limitations to Address with PyTorch
1. **Payload Classifier**: TF-IDF loses sequence patterns
2. **URL Analysis**: Only 10 features, misses character-level attacks
3. **Time-series**: Simple z-score, no learned temporal patterns
4. **Ensemble**: Linear meta-classifier, can't learn complex interactions

---

## Part 2: New PyTorch Components

### Component 1: Payload CNN
**Purpose**: Detect SQL injection, XSS, command injection from raw text

```python
class PayloadCNN(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=128, max_len=500):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.embedding(x)           # (batch, seq, embed)
        x = x.permute(0, 2, 1)          # (batch, embed, seq)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)    # (batch, 256)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return torch.sigmoid(self.fc2(x))
```

**Specs**: ~500K params | 5ms inference | 95%+ accuracy target

### Component 2: Character-Level URL CNN
**Purpose**: Detect homograph attacks, encoded payloads, suspicious patterns

```python
class URLCNN(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 64, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64 * 3, 64)
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        pooled = [self.pool(c).squeeze(-1) for c in conv_outs]
        x = torch.cat(pooled, dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
```

**Specs**: ~200K params | 3ms inference | 97%+ accuracy target

### Component 3: Time-Series LSTM
**Purpose**: Learn normal traffic patterns, detect temporal anomalies

```python
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Last timestep
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
```

**Specs**: ~100K params | 8ms inference | 90%+ detection target

### Component 4: Neural Meta-Classifier
**Purpose**: Learn optimal combination of all model outputs

```python
class MetaClassifier(nn.Module):
    def __init__(self, num_inputs=5):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
```

**Specs**: ~1K params | 1ms inference


---

## Part 3: RTX 3050 Optimization

### GPU Memory Management
```python
import torch

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Clear cache when needed
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Mixed Precision Training (FP16)
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast():  # FP16 forward pass
        output = model(batch['input'].to(device))
        loss = criterion(output, batch['target'].to(device))
    
    scaler.scale(loss).backward()  # Scaled backward
    scaler.step(optimizer)
    scaler.update()
```

### DataLoader Optimization
```python
train_loader = DataLoader(
    dataset,
    batch_size=64,           # Safe for 4GB VRAM
    shuffle=True,
    num_workers=4,           # Parallel loading
    pin_memory=True,         # Faster CPU→GPU transfer
    persistent_workers=True  # Keep workers alive
)
```

### PyTorch 2.0 Compilation (Optional Speedup)
```python
# Up to 2x inference speedup
model = torch.compile(model, mode='reduce-overhead')
```

### Training Configuration
```python
CONFIG = {
    'batch_size': 64,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 50,
    'early_stopping_patience': 5,
    'mixed_precision': True,
    'gradient_clip': 1.0
}
```

---

## Part 4: System Architecture

### Inference Pipeline
```
                         ┌─────────────────────────────────┐
                         │          Input Data             │
                         └───────────────┬─────────────────┘
                                         │
          ┌──────────────────────────────┼──────────────────────────────┐
          │                              │                              │
          ▼                              ▼                              ▼
   ┌─────────────┐              ┌─────────────────┐            ┌──────────────┐
   │ CPU Branch  │              │   GPU Branch    │            │  GPU Branch  │
   │ (sklearn)   │              │   (PyTorch)     │            │  (PyTorch)   │
   ├─────────────┤              ├─────────────────┤            ├──────────────┤
   │ Network RF  │              │ Payload CNN     │            │ URL CNN      │
   │ Fraud XGB   │              │ TimeSeries LSTM │            │              │
   │ Anomaly IF  │              │                 │            │              │
   │ URL LightGBM│              │                 │            │              │
   └──────┬──────┘              └────────┬────────┘            └──────┬───────┘
          │                              │                            │
          └──────────────────────────────┼────────────────────────────┘
                                         ▼
                         ┌─────────────────────────────────┐
                         │   Neural Meta-Classifier (GPU)  │
                         └───────────────┬─────────────────┘
                                         ▼
                         ┌─────────────────────────────────┐
                         │      Final Prediction           │
                         └─────────────────────────────────┘
```

### Latency Budget (<100ms target)
| Component | Time |
|-----------|------|
| Preprocessing | 5ms |
| sklearn models (CPU, parallel) | 10ms |
| PyTorch models (GPU, batched) | 18ms |
| Meta-classifier | 2ms |
| Overhead | 5ms |
| **Total** | **~40ms** ✓ |

### File Structure
```
src/
├── torch_models/                   # NEW
│   ├── __init__.py
│   ├── payload_cnn.py
│   ├── url_cnn.py
│   ├── timeseries_lstm.py
│   ├── meta_classifier.py
│   ├── datasets.py                 # PyTorch Dataset classes
│   └── utils.py                    # Device setup, training utils
│
├── training/                       # NEW
│   ├── train_payload.py
│   ├── train_url.py
│   ├── train_timeseries.py
│   ├── train_meta.py
│   └── trainer.py                  # Shared training loop
│
├── ensemble.py                     # MODIFY
├── predict.py                      # MODIFY
└── model_router.py                 # MODIFY

models/
├── network_intrusion_model.pkl     # Existing sklearn
├── fraud_detection_model.pkl       # Existing XGBoost
├── url_analysis_model.pkl          # Existing LightGBM
├── payload_cnn.pt                  # NEW PyTorch
├── url_cnn.pt                      # NEW PyTorch
├── timeseries_lstm.pt              # NEW PyTorch
└── meta_classifier.pt              # NEW PyTorch
```


---

## Part 5: Data Preparation

### PyTorch Dataset Classes

```python
# datasets.py
class PayloadDataset(Dataset):
    def __init__(self, texts, labels, max_len=500):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Convert to character indices (0-255)
        chars = [ord(c) % 256 for c in text[:self.max_len]]
        # Pad to max_len
        chars += [0] * (self.max_len - len(chars))
        return {
            'input': torch.tensor(chars, dtype=torch.long),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class URLDataset(Dataset):
    def __init__(self, urls, labels, max_len=200):
        self.urls = urls
        self.labels = labels
        self.max_len = max_len
    
    def __getitem__(self, idx):
        url = self.urls[idx]
        chars = [ord(c) % 128 for c in url[:self.max_len]]
        chars += [0] * (self.max_len - len(chars))
        return {
            'input': torch.tensor(chars, dtype=torch.long),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # (N, timesteps, features)
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.sequences[idx], dtype=torch.float32),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }
```

### Data Sources
| Dataset | Source | Processing |
|---------|--------|------------|
| Payloads | `datasets/security_payloads/injection/`, `fuzzing/` | Char tokenize, pad 500 |
| Benign text | `wordlists/`, synthetic | Balance with malicious |
| URLs | `datasets/url_analysis/*/` | Char tokenize, pad 200 |
| Time-series | CICIDS2017 timestamps | 60-step windows |

---

## Part 6: Training Utilities

### Early Stopping
```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop
```

### Training Loop
```python
def train_model(model, train_loader, val_loader, config):
    device = torch.device('cuda')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), 
                                   lr=config['learning_rate'],
                                   weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    criterion = nn.BCELoss()
    scaler = GradScaler()
    early_stop = EarlyStopping(patience=config['early_stopping_patience'])
    
    best_model = None
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            with autocast():
                output = model(batch['input'].to(device))
                loss = criterion(output.squeeze(), batch['target'].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                with autocast():
                    output = model(batch['input'].to(device))
                    loss = criterion(output.squeeze(), batch['target'].to(device))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
        
        if early_stop(val_loss):
            print(f"Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_model)
    return model
```

### Model Export
```python
def export_model(model, example_input, save_path):
    model.eval()
    
    # TorchScript (recommended)
    scripted = torch.jit.trace(model, example_input)
    scripted.save(f"{save_path}.pt")
    
    # ONNX (cross-platform)
    torch.onnx.export(model, example_input, f"{save_path}.onnx",
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
```


---

## Part 7: Hybrid Predictor Integration

```python
class HybridPredictor:
    """Combines sklearn (CPU) and PyTorch (GPU) models."""
    
    def __init__(self, models_dir='models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = Path(models_dir)
        self._load_models()
    
    def _load_models(self):
        # sklearn models (CPU)
        self.network_rf = joblib.load(self.models_dir / 'network_intrusion_model.pkl')
        self.fraud_xgb = joblib.load(self.models_dir / 'fraud_detection_model.pkl')
        self.url_lgbm = joblib.load(self.models_dir / 'url_analysis_model.pkl')
        
        # PyTorch models (GPU)
        self.payload_cnn = torch.jit.load(self.models_dir / 'payload_cnn.pt').to(self.device).eval()
        self.url_cnn = torch.jit.load(self.models_dir / 'url_cnn.pt').to(self.device).eval()
        self.ts_lstm = torch.jit.load(self.models_dir / 'timeseries_lstm.pt').to(self.device).eval()
        self.meta = torch.jit.load(self.models_dir / 'meta_classifier.pt').to(self.device).eval()
    
    @torch.no_grad()
    def predict(self, data):
        # sklearn predictions (CPU) - can run in parallel
        network_prob = self.network_rf['model'].predict_proba(
            self.network_rf['scaler'].transform(data['network_features'])
        )[:, 1]
        
        fraud_prob = self.fraud_xgb['model'].predict_proba(
            self.fraud_xgb['scaler'].transform(data['fraud_features'])
        )[:, 1]
        
        url_lgbm_prob = self.url_lgbm['model'].predict_proba(
            self.url_lgbm['scaler'].transform(data['url_features'])
        )[:, 1]
        
        # PyTorch predictions (GPU)
        with torch.cuda.amp.autocast():
            payload_input = torch.tensor(data['payload_chars'], dtype=torch.long).to(self.device)
            payload_prob = self.payload_cnn(payload_input).cpu().numpy().flatten()
            
            url_input = torch.tensor(data['url_chars'], dtype=torch.long).to(self.device)
            url_cnn_prob = self.url_cnn(url_input).cpu().numpy().flatten()
            
            ts_input = torch.tensor(data['timeseries'], dtype=torch.float32).to(self.device)
            ts_prob = self.ts_lstm(ts_input).cpu().numpy().flatten()
        
        # Hybrid URL score
        url_prob = 0.6 * url_lgbm_prob + 0.4 * url_cnn_prob
        
        # Meta-classifier combines all
        meta_input = torch.tensor(
            np.column_stack([network_prob, fraud_prob, url_prob, payload_prob, ts_prob]),
            dtype=torch.float32
        ).to(self.device)
        
        with torch.cuda.amp.autocast():
            final_prob = self.meta(meta_input).cpu().numpy().flatten()
        
        return {
            'is_attack': (final_prob > 0.5).astype(int),
            'confidence': final_prob,
            'scores': {
                'network': network_prob,
                'fraud': fraud_prob,
                'url': url_prob,
                'payload': payload_prob,
                'timeseries': ts_prob
            }
        }
```

---

## Part 8: Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] Install PyTorch with CUDA
- [ ] Create `src/torch_models/utils.py`
- [ ] Verify GPU detection and memory
- [ ] Create Dataset classes
- [ ] Benchmark GPU performance

**Verification**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### Phase 2: Payload CNN (Week 2)
- [ ] Implement `payload_cnn.py`
- [ ] Prepare payload dataset (balance classes)
- [ ] Train with mixed precision
- [ ] Validate >95% accuracy
- [ ] Export to TorchScript

**Success**: 95%+ accuracy, <10ms inference

### Phase 3: URL CNN (Week 3)
- [ ] Implement `url_cnn.py`
- [ ] Prepare URL dataset
- [ ] Train and validate
- [ ] Create hybrid scorer (LightGBM + CNN)
- [ ] Export model

**Success**: 97%+ accuracy, <5ms inference

### Phase 4: Time-Series LSTM (Week 4)
- [ ] Generate time-series windows from CICIDS2017
- [ ] Implement `timeseries_lstm.py`
- [ ] Train on normal/attack patterns
- [ ] Validate detection rate
- [ ] Export model

**Success**: 90%+ detection, <1% false positive

### Phase 5: Meta-Classifier (Week 5)
- [ ] Collect all model outputs on validation set
- [ ] Implement `meta_classifier.py`
- [ ] Train ensemble combiner
- [ ] Calibrate probabilities
- [ ] Export model

**Success**: 97%+ overall accuracy

### Phase 6: Integration (Week 6)
- [ ] Implement `HybridPredictor` class
- [ ] Update `predict.py`
- [ ] Update `ensemble.py`
- [ ] End-to-end testing
- [ ] Latency benchmarking

**Success**: <100ms latency, all tests pass

### Phase 7: Production (Week 7)
- [ ] Add fallback to sklearn if GPU fails
- [ ] Implement monitoring/logging
- [ ] Model versioning
- [ ] Documentation
- [ ] Final validation

**Success**: Zero errors in 24hr test


---

## Part 9: Dependencies

### requirements.txt (Updated)
```
# Existing
pandas
scikit-learn
numpy
requests
matplotlib
seaborn
xgboost
lightgbm
joblib

# PyTorch Stack (NEW)
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Optional Enhancements
torchmetrics>=1.2.0       # Better metrics
onnx>=1.15.0              # Model export
onnxruntime-gpu>=1.16.0   # Fast ONNX inference
```

### Installation Commands
```bash
# Create environment
conda create -n hacking-detection python=3.11
conda activate hacking-detection

# Install PyTorch with CUDA 12.1 (for RTX 3050)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
```

---

## Part 10: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| VRAM overflow | Medium | High | Batch size 64, gradient checkpointing, `torch.cuda.empty_cache()` |
| Training instability | Low | Medium | Gradient clipping (1.0), LR scheduler, mixed precision |
| Overfitting | Medium | Medium | Dropout (0.2-0.3), early stopping, data augmentation |
| Latency regression | Low | High | Benchmark each component, TorchScript compilation |
| Accuracy regression | Low | High | A/B testing, keep sklearn fallback |
| GPU driver issues | Low | Medium | Document CUDA 12.1 requirement, test on clean install |
| Model compatibility | Low | Low | Pin PyTorch version, export to ONNX |

### Fallback Strategy
```python
class HybridPredictor:
    def predict(self, data):
        try:
            return self._predict_with_gpu(data)
        except RuntimeError as e:
            if 'CUDA' in str(e):
                logger.warning("GPU failed, falling back to CPU")
                return self._predict_cpu_only(data)
            raise
```

---

## Part 11: Success Metrics

### Accuracy Targets
| Model | Current | Target | Improvement |
|-------|---------|--------|-------------|
| Payload classifier | ~88% (TF-IDF) | 95%+ | +7% |
| URL classifier | ~94% (LightGBM) | 97%+ (hybrid) | +3% |
| Time-series detector | N/A (z-score) | 90%+ | New capability |
| Overall ensemble | ~95% | 97%+ | +2% |

### Performance Targets
| Metric | Target |
|--------|--------|
| Inference latency | <100ms |
| Training time (per model) | <30 min |
| GPU memory usage | <3.5GB |
| GPU utilization (training) | >70% |

### Operational Targets
| Metric | Target |
|--------|--------|
| Uptime | 99.9% |
| Error rate | <0.1% |
| Model reload time | <5s |
| Fallback success rate | 100% |

---

## Part 12: Quick Reference

### Training Commands
```bash
# Train all PyTorch models
python src/training/train_payload.py
python src/training/train_url.py
python src/training/train_timeseries.py
python src/training/train_meta.py
```

### Inference
```bash
# Single prediction
python src/predict.py --input sample.json

# Batch prediction
python src/predict.py --input batch.json --batch-size 64
```

### Monitoring
```python
# Check GPU memory
import torch
print(f"Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / 4GB")

# Check model latency
import time
start = time.perf_counter()
predictor.predict(sample_data)
print(f"Latency: {(time.perf_counter()-start)*1000:.1f}ms")
```

---

## Appendix A: PyTorch vs TensorFlow Comparison

| Feature | PyTorch | TensorFlow |
|---------|---------|------------|
| Execution | Eager (debug-friendly) | Graph (needs tf.function) |
| Syntax | Pythonic | More verbose |
| Debugging | Python debugger works | Harder to debug |
| Custom ops | Easy | More complex |
| Mobile deploy | TorchScript, ONNX | TFLite |
| Serving | TorchServe, FastAPI | TF Serving |
| Community | Research-focused | Production-focused |
| RTX 3050 | Excellent | Excellent |

**Verdict**: PyTorch chosen for easier debugging, cleaner code, and better fit with existing sklearn codebase.

---

## Appendix B: Model Size Summary

| Model | Parameters | Size (MB) | VRAM (inference) |
|-------|------------|-----------|------------------|
| Payload CNN | ~500K | ~2 | ~50MB |
| URL CNN | ~200K | ~1 | ~30MB |
| TimeSeries LSTM | ~100K | ~0.5 | ~20MB |
| Meta Classifier | ~1K | ~0.01 | ~5MB |
| **Total PyTorch** | ~800K | ~3.5 | ~105MB |

sklearn models run on CPU, no VRAM impact.

---

*Document created: 2025-12-19*
*Framework: PyTorch 2.1+ with CUDA 12.1*
*Next step: Begin Phase 1 - Install PyTorch and verify GPU*
