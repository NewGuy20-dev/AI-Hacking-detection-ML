# Implementation Plan: 60M Benign Data Expansion + HostBehaviorDetector

## Executive Summary

| Item | Value |
|------|-------|
| **Objective** | Expand benign dataset 5M → 60M, create HostBehaviorDetector |
| **Hardware** | NVIDIA RTX 3050 Laptop GPU (4GB VRAM) |
| **Batch Size** | 512 |
| **Total Dataset** | 151.5M samples (91.5M malicious + 60M benign) |
| **Training Time** | **5-6 hours** (Transfer Learning + Smart Sampling) |
| **Stress Test** | **30 min - 1 hour** |
| **Total Time** | **~7 hours** |

---

## Part 1: Transfer Learning Strategy

### 1.1 Why Transfer Learning?

| Approach | Training Time | Feasible? |
|----------|---------------|-----------|
| Train from scratch (all 151.5M) | ~41 hours | ❌ |
| Transfer Learning + Smart Sampling | ~5-6 hours | ✅ |

### 1.2 Transfer Learning Approaches

#### Approach A: Pre-trained Character Embeddings
| Source | Description | Speed Gain |
|--------|-------------|------------|
| CharacterBERT | Character-aware BERT embeddings | 3-4x faster |
| FastText Subword | Subword embeddings (adaptable) | 2-3x faster |
| Custom Pre-training | Self-supervised on security corpus | 2-3x faster |

#### Approach B: Frozen Embedding Strategy
```
Epoch 1-2: FREEZE embeddings → Train conv + FC only (fast convergence)
Epoch 3-5: UNFREEZE all → Fine-tune with lower LR (refinement)
```

#### Approach C: Security Domain Pre-training (Recommended)
1. Pre-train embeddings on UNLABELED security data:
   - SecLists payloads (1.8GB)
   - PayloadsAllTheThings
   - Benign code/text corpus (60M samples)
2. Task: Masked character prediction (BERT-style)
3. Then fine-tune on labeled classification task

### 1.3 Model-Specific Transfer Learning

| Model | Transfer Source | Frozen Layers | Fine-tune Strategy |
|-------|-----------------|---------------|-------------------|
| **PayloadCNN** | Pre-trained char embeddings | Embedding (epochs 1-2) | Gradual unfreeze |
| **URL CNN** | URL classification embeddings | Embedding (epochs 1-2) | Gradual unfreeze |
| **Timeseries LSTM** | Anomaly detection weights | LSTM layers (epoch 1) | Full fine-tune |

### 1.4 Smart Sampling Strategy

Instead of training on ALL 151.5M samples every epoch:

| Strategy | Samples/Epoch | Composition | Benefit |
|----------|---------------|-------------|---------|
| **Balanced Sampling** | 20M | 10M malicious + 10M benign | Fast + balanced |
| **Stratified** | 20M | Proportional to class sizes | Preserves distribution |
| **Curriculum** | 10M → 30M | Easy → hard samples | Better convergence |

**Key**: Different random samples each epoch → model sees variety without excessive time

---

## Part 2: Training Time Budget (5-6 Hours)

### 2.1 Detailed Time Breakdown

| Model | Samples/Epoch | Batches/Epoch | Time/Epoch | Epochs | Total |
|-------|---------------|---------------|------------|--------|-------|
| **PayloadCNN** | 20M | 39,062 | ~33 min | 5 | **2.75 hrs** |
| **URL CNN** | 10M | 19,531 | ~16 min | 5 | **1.33 hrs** |
| **Timeseries LSTM** | 5M | 9,766 | ~8 min | 5 | **0.67 hrs** |
| **Subtotal** | - | - | - | - | **4.75 hrs** |
| **Buffer/Overhead** | - | - | - | - | **+1 hr** |
| **TOTAL TRAINING** | - | - | - | - | **5-6 hrs** |

### 2.2 Training Schedule (Hour by Hour)

```
┌────────────────────────────────────────────────────────────────┐
│ TRAINING TIMELINE (5-6 hours)                                  │
├────────────────────────────────────────────────────────────────┤
│ Hour 0.0 - 0.5:  Setup, data loading, GPU warmup               │
│ Hour 0.5 - 3.25: PayloadCNN training (5 epochs)                │
│   ├── Epoch 1: Frozen embeddings, LR=1e-3                      │
│   ├── Epoch 2: Frozen embeddings, LR=1e-3                      │
│   ├── Epoch 3: Unfreeze all, LR=5e-4                           │
│   ├── Epoch 4: Full training, LR=2e-4                          │
│   └── Epoch 5: Full training, LR=1e-4                          │
│ Hour 3.25 - 4.5: URL CNN training (5 epochs)                   │
│   ├── Epoch 1-2: Frozen embeddings                             │
│   └── Epoch 3-5: Full fine-tuning                              │
│ Hour 4.5 - 5.25: Timeseries LSTM training (5 epochs)           │
│   ├── Epoch 1: Frozen LSTM, train FC only                      │
│   └── Epoch 2-5: Full fine-tuning                              │
│ Hour 5.25 - 6.0: Buffer for issues, final saves                │
└────────────────────────────────────────────────────────────────┘
```

### 2.3 Learning Rate Schedule with Transfer Learning

```
PayloadCNN Learning Rate:
┌─────────────────────────────────────────────────────────────┐
│ LR │                                                        │
│1e-3│ ████████████████                                       │ Epochs 1-2 (frozen)
│5e-4│                 ████████                               │ Epoch 3 (unfreeze)
│2e-4│                         ████████                       │ Epoch 4
│1e-4│                                 ████████               │ Epoch 5
│    └────────────────────────────────────────────────────────│
│         Epoch 1    Epoch 2    Epoch 3    Epoch 4    Epoch 5 │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 3: Stress Test & Validation Plan (30 min - 1 hour)

### 3.1 Test Phases Overview

| Phase | Duration | Description |
|-------|----------|-------------|
| **Phase 1** | 10 min | Quick validation on holdout set |
| **Phase 2** | 15 min | Adversarial testing |
| **Phase 3** | 10 min | Performance benchmarking |
| **Phase 4** | 15 min | False positive analysis |
| **Phase 5** | 10 min | Report generation |
| **TOTAL** | **60 min** | |

### 3.2 Phase 1: Quick Validation (10 minutes)

| Test | Samples | Metrics | Pass Criteria |
|------|---------|---------|---------------|
| Holdout accuracy | 1.5M | Accuracy | ≥98.9% |
| Per-class metrics | 1.5M | Precision, Recall | ≥98% each |
| Confidence calibration | 100K | ECE | <0.05 |
| Class balance check | 1.5M | Distribution | Within 5% of expected |

### 3.3 Phase 2: Adversarial Testing (15 minutes)

| Test | Description | Samples | Pass Criteria |
|------|-------------|---------|---------------|
| Evasion attacks | Obfuscated payloads | 10K | >95% detection |
| Encoding bypass | URL/HTML/Base64 encoding | 10K | >90% detection |
| Case variations | Upper/lower/mixed case | 10K | Consistent ±2% |
| Whitespace injection | Extra spaces/tabs | 10K | >95% detection |
| Comment injection | SQL/HTML comments | 10K | >90% detection |
| Unicode tricks | Homoglyphs, zero-width | 5K | >85% detection |

### 3.4 Phase 3: Performance Benchmarking (10 minutes)

| Metric | Target | Test Method |
|--------|--------|-------------|
| Single inference latency | <100ms | 1000 single samples |
| Batch throughput | >5000 samples/sec | Batch 512, 100 iterations |
| GPU memory (inference) | <2GB | nvidia-smi monitoring |
| GPU memory (training) | <3.5GB | nvidia-smi monitoring |
| CPU fallback latency | <500ms | Force CPU, 100 samples |
| Cold start time | <5s | Model load time |

### 3.5 Phase 4: False Positive Analysis (15 minutes)

| Test Category | Samples | Target FP Rate | Action if Fail |
|---------------|---------|----------------|----------------|
| Benign URLs | 5M (sampled) | <2% | Review threshold |
| Benign payloads | 5M (sampled) | <3% | Add to training |
| Code snippets | 500K | <5% | Context classifier |
| Natural language | 500K | <2% | Adjust weights |
| SQL queries (legit) | 100K | <3% | Pattern whitelist |
| Shell commands (safe) | 100K | <3% | Pattern whitelist |

### 3.6 Phase 5: Report Generation (10 minutes)

**Output Structure:**
```
stress_test_report_YYYYMMDD_HHMMSS/
├── summary.json              # Overall PASS/FAIL + key metrics
├── detailed_metrics.json     # All metrics with breakdowns
├── false_positives.csv       # FP samples for manual review
├── false_negatives.csv       # FN samples for manual review
├── performance_benchmark.json # Latency/throughput results
├── adversarial_results.json  # Adversarial test results
├── calibration_plot.png      # Reliability diagram
├── confusion_matrix.png      # Per-class confusion
├── roc_curves.png            # ROC curves per model
└── report.html               # Human-readable summary
```

**Summary JSON Format:**
```json
{
  "timestamp": "2025-12-23T18:00:00Z",
  "overall_status": "PASS",
  "training_time_hours": 5.5,
  "models_trained": ["payload_cnn", "url_cnn", "timeseries_lstm"],
  "metrics": {
    "accuracy": 0.991,
    "precision": 0.988,
    "recall": 0.994,
    "f1": 0.991,
    "fp_rate": 0.024,
    "fn_rate": 0.018
  },
  "performance": {
    "inference_latency_ms": 45,
    "throughput_samples_per_sec": 6200,
    "gpu_memory_mb": 1850
  },
  "tests_passed": 42,
  "tests_failed": 0
}
```

---

## Part 4: RTX 3050 Hardware Optimization

### 4.1 Specifications

| Spec | Value |
|------|-------|
| VRAM | 4GB GDDR6 |
| CUDA Cores | 2048 |
| Memory Bandwidth | 192 GB/s |
| TDP | 35-80W |

### 4.2 Memory Budget (Batch Size 512)

```
┌─────────────────────────────────────────────────────────────┐
│ VRAM ALLOCATION (Batch Size 512)                            │
├─────────────────────────────────────────────────────────────┤
│ PayloadCNN Model Weights:     ~3 MB                         │
│ Input Batch (512 × 500):      ~1 MB                         │
│ Embeddings (FP16):            ~64 MB                        │
│ Conv Activations:             ~100 MB                       │
│ Gradients:                    ~100 MB                       │
│ Optimizer States (Adam):      ~6 MB                         │
│ ─────────────────────────────────────────────────────────── │
│ TOTAL PEAK:                   ~300-400 MB                   │
│ AVAILABLE:                    4000 MB                       │
│ HEADROOM:                     ~3600 MB ✅                   │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Optimization Techniques

| Technique | Description | Memory Savings |
|-----------|-------------|----------------|
| **Mixed Precision (AMP)** | FP16 forward/backward | ~50% |
| **Gradient Checkpointing** | Recompute activations | ~30% (if needed) |
| **Pin Memory** | Faster CPU→GPU transfer | N/A (speed) |
| **Async Data Loading** | 4 workers, prefetch | N/A (speed) |

---

## Part 5: Dataset Expansion Strategy (5M → 60M Benign)

### 5.1 Current vs Target State

| Category | Current | Target | Change |
|----------|---------|--------|--------|
| Malicious | 91.5M | 91.5M | - |
| Benign | 5M | 60M | **+55M** |
| **Total** | 96.5M | 151.5M | **+55M** |
| Ratio (B:M) | 1:18 | 1:1.5 | **Balanced** |

### 5.2 Benign Data Sources (60M Total)

| Source | Count | Method | Priority |
|--------|-------|--------|----------|
| **URL Variations** | 20M | 20 variations per Tranco domain | HIGH |
| **CICIDS2017 Benign** | 5M | Extract benign network flows | HIGH |
| **UNSW-NB15 Benign** | 5M | Extract benign network flows | HIGH |
| **Legitimate SQL** | 5M | Template generation | HIGH |
| **Safe Shell Commands** | 3M | Template generation | MEDIUM |
| **API Calls/JSON** | 5M | REST/GraphQL templates | MEDIUM |
| **Code Snippets** | 5M | Multi-language (Python, JS, Java) | MEDIUM |
| **Log Entries** | 3M | Apache/syslog/JSON formats | MEDIUM |
| **Config Files** | 2M | YAML/JSON/XML/ENV | LOW |
| **Email/Chat** | 2M | Faker + templates | LOW |
| **Existing Expansion** | 5M | Expand curated benign | LOW |
| **TOTAL** | **60M** | | |

### 5.3 Data Quality Requirements

| Requirement | Description | Validation |
|-------------|-------------|------------|
| No duplicates | Hash-based deduplication | MurmurHash3 |
| No overlap with malicious | Check against attack patterns | Pattern matching |
| Realistic distribution | Match real-world frequencies | Statistical tests |
| Proper encoding | UTF-8, handle special chars | Encoding validation |
| Balanced categories | ~equal samples per category | Count verification |

---

## Part 6: HostBehaviorDetector Design

### 6.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  HostBehaviorDetector                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Syscall     │  │ Memory      │  │ Process     │         │
│  │ LSTM        │  │ CNN         │  │ Rules       │         │
│  │ (ADFA-IDS)  │  │ (MalMem)    │  │ (EVTX)      │         │
│  │ Weight: 0.3 │  │ Weight: 0.3 │  │ Weight: 0.2 │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ↓                                  │
│              ┌─────────────────────┐                        │
│              │  File Anomaly       │                        │
│              │  Detector           │                        │
│              │  Weight: 0.2        │                        │
│              └──────────┬──────────┘                        │
│                         ↓                                   │
│              ┌─────────────────────┐                        │
│              │  Weighted Ensemble  │                        │
│              │  → Final Score      │                        │
│              │  → Attack Type      │                        │
│              │  → Indicators       │                        │
│              └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Integration with Main Ensemble

**Current Weights:**
```
Network: 0.50, URL: 0.30, Content: 0.20
```

**New Weights (with Host):**
```
Network: 0.35, URL: 0.25, Content: 0.15, Host: 0.25
```

### 6.3 Attack Type Mapping

| Detection Source | Attack Types |
|------------------|--------------|
| Syscall LSTM | Process injection, Privilege escalation, Reverse shell |
| Memory CNN | Rootkit, Code injection, Memory corruption |
| Process Rules | Macro execution, LOLBins, Encoded commands |
| File Anomaly | Persistence, Data exfiltration, Ransomware |

---

## Part 7: Implementation Timeline

### 7.1 Day-by-Day Schedule

| Day | Phase | Tasks | Duration |
|-----|-------|-------|----------|
| **Day 1** | Data Prep | Generate 60M benign samples | 8 hrs |
| **Day 2** | Data Prep | Validate, deduplicate, split | 4 hrs |
| **Day 2** | Infra | Set up training pipeline | 4 hrs |
| **Day 3** | Training | Train all models (transfer learning) | **5-6 hrs** |
| **Day 3** | Validation | Stress test & validation | **1 hr** |
| **Day 4** | Host Agent | Implement HostBehaviorDetector | 6 hrs |
| **Day 5** | Integration | Integrate with ensemble, final tests | 4 hrs |

### 7.2 Detailed Task Checklist

#### Day 1-2: Data Generation
- [ ] Create `generate_60m_benign.py` script
- [ ] Extract CICIDS2017 benign flows (5M)
- [ ] Extract UNSW-NB15 benign flows (5M)
- [ ] Generate URL variations from Tranco (20M)
- [ ] Generate legitimate SQL queries (5M)
- [ ] Generate safe shell commands (3M)
- [ ] Generate API calls and JSON (5M)
- [ ] Generate code snippets (5M)
- [ ] Generate log entries (3M)
- [ ] Generate config files (2M)
- [ ] Generate email/chat content (2M)
- [ ] Expand existing benign (5M)
- [ ] Deduplicate all data
- [ ] Validate no overlap with malicious

#### Day 2: Training Infrastructure
- [ ] Create `StreamingDataset` class
- [ ] Create `train_rtx3050.py` script
- [ ] Implement transfer learning logic
- [ ] Implement checkpoint manager
- [ ] Implement smart sampling
- [ ] Test on small subset (1M samples)

#### Day 3: Training (5-6 hours)
- [ ] Pre-train/load embeddings
- [ ] Train PayloadCNN (2.75 hrs)
- [ ] Train URL CNN (1.33 hrs)
- [ ] Train Timeseries LSTM (0.67 hrs)
- [ ] Save best models

#### Day 3: Stress Test (1 hour)
- [ ] Run quick validation (10 min)
- [ ] Run adversarial tests (15 min)
- [ ] Run performance benchmarks (10 min)
- [ ] Run FP analysis (15 min)
- [ ] Generate reports (10 min)

#### Day 4: HostBehaviorDetector
- [ ] Create `host_behavior_detector.py`
- [ ] Implement syscall analyzer (LSTM)
- [ ] Implement memory analyzer (CNN)
- [ ] Implement process rule engine
- [ ] Implement file anomaly detector
- [ ] Unit tests

#### Day 5: Integration
- [ ] Update ensemble weights
- [ ] Integration tests
- [ ] End-to-end validation
- [ ] Update documentation
- [ ] Create deployment package

---

## Part 8: File Deliverables

```
NEW FILES TO CREATE:
────────────────────

scripts/
├── generate_60m_benign.py       # Benign data expansion
├── train_rtx3050.py             # RTX 3050 optimized training
├── stress_test.py               # Comprehensive stress testing
└── validate_models.py           # Model validation

src/
├── agents/
│   └── host_behavior_detector.py  # HostBehaviorDetector agent
├── data/
│   ├── streaming_dataset.py       # Memory-efficient data loading
│   ├── benign_generators.py       # Benign data generators
│   └── smart_sampler.py           # Balanced sampling logic
└── training/
    ├── transfer_learning.py       # Transfer learning utilities
    └── trainer_rtx3050.py         # Training orchestrator

configs/
├── training_rtx3050.yaml          # Training configuration
├── stress_test_config.yaml        # Stress test configuration
└── host_detector_config.yaml      # Host detector configuration

datasets/
├── benign_60m/                    # Generated benign data
│   ├── urls_20m.jsonl             # ~2GB
│   ├── network_10m.jsonl          # ~1GB
│   ├── sql_5m.jsonl               # ~500MB
│   ├── shell_3m.jsonl             # ~300MB
│   ├── api_5m.jsonl               # ~500MB
│   ├── code_5m.jsonl              # ~600MB
│   ├── logs_3m.jsonl              # ~400MB
│   ├── config_2m.jsonl            # ~200MB
│   ├── email_2m.jsonl             # ~250MB
│   └── expanded_5m.jsonl          # ~500MB
└── host_based/                    # Host-based datasets
    ├── adfa_ids/                  # ~500MB
    ├── darpa/                     # ~221MB
    ├── cic_malmem/                # ~1.2GB
    └── evtx_samples/              # ~300MB
```

---

## Part 9: Success Criteria

| Metric | Target | Priority |
|--------|--------|----------|
| **Accuracy** | ≥98.9% | CRITICAL |
| **Recall** | ≥98% | CRITICAL |
| **FP Rate** | ≤3% | CRITICAL |
| **FN Rate** | ≤2% | CRITICAL |
| **Training Time** | ≤6 hours | HIGH |
| **Stress Test Time** | ≤1 hour | HIGH |
| **Peak VRAM** | ≤3.5GB | HIGH |
| **Inference Latency** | ≤100ms | MEDIUM |
| **Throughput** | ≥5000/sec | MEDIUM |

---

## Part 10: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OOM on RTX 3050 | Low | High | Reduce batch to 256 + gradient accumulation |
| Training exceeds 6 hrs | Medium | Medium | Reduce epochs, increase sampling |
| Transfer learning fails | Low | High | Fall back to training from scratch |
| Data quality issues | Medium | High | Validate samples, check overlap |
| Overfitting | Medium | High | Early stopping, validation monitoring |
| Disk space (~15GB) | Low | Low | Auto-delete old checkpoints on retrain |

---

## Part 11: Disk Space & Checkpoint Policy

### Storage Breakdown
| Category | Size | Permanent? |
|----------|------|------------|
| New benign data (60M) | ~7GB | Yes |
| Host-based datasets | ~2.2GB | Yes |
| Final model weights | ~50MB | Yes |
| Checkpoints (during training) | ~4.5GB | No |
| Temp/intermediate files | ~3GB | No |
| **Peak during training** | **~17GB** | |
| **Permanent storage** | **~10GB** | |

### Checkpoint Auto-Delete Policy
```
DURING TRAINING:
  - Keep last 3 checkpoints per model
  - Save checkpoint every 10,000 batches

AFTER TRAINING COMPLETE:
  - Delete ALL checkpoints
  - Keep only final model weights (.pt and .pth files)
```

---

## Part 12: Commands Reference

### Generate Benign Data
```bash
python scripts/generate_60m_benign.py \
    --output datasets/benign_60m/ \
    --tranco datasets/url_analysis/top-1m.csv \
    --cicids datasets/network_intrusion/cicids2017/ \
    --unsw datasets/network_intrusion/unsw_nb15/
```

### Train Models (5-6 hours)
```bash
# Train all models with transfer learning
python scripts/train_rtx3050.py \
    --config configs/training_rtx3050.yaml \
    --transfer-learning \
    --samples-per-epoch 20000000

# Train specific model
python scripts/train_rtx3050.py \
    --model payload_cnn \
    --epochs 5 \
    --freeze-epochs 2

# Resume from checkpoint
python scripts/train_rtx3050.py \
    --resume checkpoints/payload_cnn_latest.pt
```

### Run Stress Test (30 min - 1 hour)
```bash
python scripts/stress_test.py \
    --config configs/stress_test_config.yaml \
    --output reports/stress_test_$(date +%Y%m%d)/
```

### Validate Models
```bash
python scripts/validate_models.py \
    --models models/ \
    --holdout datasets/holdout_test/ \
    --benign datasets/benign_60m/
```

---

**Document Version**: 2.0
**Last Updated**: December 23, 2025
**Key Changes**: Added transfer learning, reduced training time to 5-6 hours, added stress test plan
