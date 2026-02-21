# Training & Stress Test Pipeline Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install mmh3
```

### 2. Run Commands

**Full Pipeline (Train + Stress Test):**
```bash
python src/stress_test/train_pipeline.py --full
```

**Training Only:**
```bash
python src/stress_test/train_pipeline.py --train --epochs 10
```

**Stress Test Only (uses existing model):**
```bash
python src/stress_test/runner.py
```

---

## What Each Command Does

### Full Pipeline (`--full`)
1. Loads 96M samples via streaming (no RAM issues)
2. Trains with mixed precision (2-3x faster)
3. Saves model + hash registry
4. Runs stress test with fresh data
5. Sends Discord report with full metrics

### Training Only (`--train`)
1. Streams training data
2. Mixed precision training
3. Early stopping if validation plateaus
4. Saves best model checkpoint

### Stress Test Only
1. Generates 50% adversarial + 50% scraped fresh data
2. Verifies NO overlap with training data
3. Runs inference + performance tests
4. Calculates accuracy, FP rate, FN rate, calibration
5. Sends Discord report

---

## Discord Notifications

All commands automatically send results to Discord.

### Success Message
```
✅ DAILY STRESS TEST PASSED
┌────────────────┬─────────┬────────┬────────┐
│ Metric         │ Value   │ Target │ Status │
├────────────────┼─────────┼────────┼────────┤
│ Accuracy       │ 99.87%  │ >99%   │   ✅   │
│ FP Rate        │  1.20%  │ <3%    │   ✅   │
│ FN Rate        │  0.80%  │ <2%    │   ✅   │
│ P95 Latency    │   45ms  │ <100ms │   ✅   │
└────────────────┴─────────┴────────┴────────┘
```

### Failure Message
```
🚨 STRESS TEST FAILED
- Shows which metrics failed
- Lists failed sample examples
- Provides action recommendations
```

---

## Options

| Flag | Description |
|------|-------------|
| `--epochs N` | Number of training epochs (default: 10) |
| `--batch-size N` | Batch size (default: 256) |
| `--model PATH` | Model save path (default: models/payload_cnn.pt) |
| `--no-discord` | Skip Discord notification |
| `--no-perf` | Skip performance tests |

---

## Reports

Reports are saved to `reports/` directory:
- `stress_test_YYYY-MM-DD.json` - Machine-readable
- `stress_test_YYYY-MM-DD.txt` - Human-readable

---

## Typical Workflow

1. **After adding new training data:**
   ```bash
   python src/stress_test/train_pipeline.py --full
   ```

2. **Daily check (no retraining):**
   ```bash
   python src/stress_test/runner.py
   ```

3. **Quick training test:**
   ```bash
   python src/stress_test/train_pipeline.py --train --epochs 3
   ```

---

## Troubleshooting

**Out of memory:**
- Reduce batch size: `--batch-size 128`

**Slow training:**
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

**Discord not working:**
- Check webhook URL in alerting.py
- Ensure `requests` is installed

---

## File Structure

```
src/stress_test/
├── train_pipeline.py    # Main entry point
├── runner.py            # Stress test runner
├── amp_trainer.py       # Mixed precision training
├── streaming_dataset.py # Handles 96M samples
├── hash_registry.py     # Prevents data leakage
├── metrics.py           # Accuracy, FP, FN calculation
├── alerting.py          # Discord notifications
├── reporter.py          # Report generation
├── generators/
│   ├── adversarial.py   # Attack variations
│   ├── edge_cases.py    # Unicode, long inputs
│   └── scrapers.py      # Fresh web data
└── suites/
    └── performance_suite.py  # Latency tests
```
