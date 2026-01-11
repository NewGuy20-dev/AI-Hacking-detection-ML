# Enhanced train_rtx3050.py - Complete Training Script

## ✅ What Was Added:

### 1. **All 6 Models Support**
- ✅ Payload CNN (PyTorch)
- ✅ URL CNN (PyTorch)
- ✅ Timeseries LSTM (PyTorch)
- ✅ Network Intrusion RF (sklearn) - calls existing script
- ✅ Fraud Detection XGBoost (sklearn) - calls existing script
- ✅ Host Behavior RF (sklearn) - calls existing script

### 2. **Specialized Data Mapping**
Shows exact data for each model before training:
```
[1/6] PAYLOAD CNN
  Description: Character-level CNN for injection attack detection
  Malicious: 94.5M security payloads (SQL, XSS, command injection)
  Benign: 192M curated + live benign
  Total: ~286M samples
```

### 3. **Dataset Statistics**
Shows at startup:
```
📊 DATASET OVERVIEW
  Total: 322.6M samples (54.51 GB)
  Benign: 215.6M (66.8%)
  Malicious: 107.1M (33.2%)
  Ratio: 2:1 (optimal for production)
```

### 4. **Enhanced Main Function**
- Supports all 6 models
- Can train: all, pytorch, sklearn, or individual models
- Shows comprehensive summary with success/fail status

## 📋 Usage:

```powershell
# Train all 6 models
python scripts/train_rtx3050.py --model all

# Train only PyTorch models (payload, url, timeseries)
python scripts/train_rtx3050.py --model pytorch

# Train only sklearn models (network, fraud, host)
python scripts/train_rtx3050.py --model sklearn

# Train specific model
python scripts/train_rtx3050.py --model payload
python scripts/train_rtx3050.py --model network

# Resume PyTorch training
python scripts/train_rtx3050.py --model payload --resume

# Custom hyperparameters
python scripts/train_rtx3050.py --model all --epochs 10 --batch-size 512
```

## 🔄 What Happens:

### For PyTorch Models (payload, url, timeseries):
1. Shows data mapping
2. Loads streaming dataset (handles 286M samples)
3. Transfer learning (freeze → unfreeze)
4. Checkpoints every 1000 batches
5. Can resume from any batch
6. Saves to `models/payload_cnn.pt`

### For Sklearn Models (network, fraud, host):
1. Shows data mapping
2. Directs you to run existing training scripts:
   - `python src/train_network_intrusion.py`
   - `python src/train_fraud_detection.py`
   - `python src/train_host_behavior.py`

## 📊 Output Example:

```
================================================================================
 COMPLETE MODEL RETRAINING PIPELINE
 Started: 2026-01-09 18:30:00
================================================================================

📊 DATASET OVERVIEW
  Total: 322.6M samples (54.51 GB)
  Benign: 215.6M (66.8%)
  Malicious: 107.1M (33.2%)
  Ratio: 2:1 (optimal for production)
================================================================================

GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4.3 GB)

[1/6] PAYLOAD CNN
--------------------------------------------------------------------------------
  Description: Character-level CNN for injection attack detection
  Malicious: 94.5M security payloads (SQL, XSS, command injection)
  Benign: 192M curated + live benign
  Total: ~286M samples
--------------------------------------------------------------------------------
Training...

================================================================================
TRAINING SUMMARY
================================================================================
Total time: 13:45:23
Successful: 6/6
Failed: 0/6

Model Status:
  PAYLOAD CNN                   : ✓ SUCCESS
  URL CNN                       : ✓ SUCCESS
  TIMESERIES LSTM               : ✓ SUCCESS
  NETWORK INTRUSION RF          : ✓ SUCCESS
  FRAUD DETECTION XGBOOST       : ✓ SUCCESS
  HOST BEHAVIOR RF              : ✓ SUCCESS
================================================================================
```

## 🎯 Key Features Preserved:

- ✅ Streaming datasets (handles 286M samples)
- ✅ Transfer learning (freeze/unfreeze embeddings)
- ✅ Checkpointing (every 1000 batches)
- ✅ Resume capability (from any batch)
- ✅ Mixed precision training (AMP)
- ✅ Discord notifications
- ✅ Optimized for RTX 3050 (1024 batch, 6 workers)

## 📁 Files:

- `train_rtx3050.py` - Enhanced version (current)
- `train_rtx3050_original.py` - Original backup
- `train_rtx3050_backup.py` - Another backup

## ⏱️ Estimated Training Time:

| Model | Time | GPU/CPU |
|-------|------|---------|
| Payload CNN | 8h | GPU |
| URL CNN | 3h | GPU |
| Timeseries LSTM | 1h | GPU |
| Network RF | 15m | CPU |
| Fraud XGBoost | 10m | CPU |
| Host RF | 10m | CPU |
| **TOTAL** | **~13h** | |

## 🚀 Ready to Train!

Run: `python scripts/train_rtx3050.py --model all`
