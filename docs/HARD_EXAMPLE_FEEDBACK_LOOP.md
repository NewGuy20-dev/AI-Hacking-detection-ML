# Hard-Example Feedback Loop (Payload + URL)

## Goal
Continuously improve `payload` and `url` models by learning from failed stress-test scenarios using a manual, strict-gated retraining loop.

## Entry Command
- Dry-run analysis only:
  - `py -3 src/feedback_loop/hard_example_loop.py --model payload,url --dry-run`
- Full retrain with strict gate check (manual promotion attempt):
  - `py -3 src/feedback_loop/hard_example_loop.py --model payload,url --promote`

## Flow
1. Ingest failures from `evaluation/stress_test_v14/*_failures.jsonl`.
2. Generate category-preserving hard variants.
3. Build replay dataset from baseline + prior hard examples + new hard examples.
4. Retrain payload/url with `--hard-examples-file`.
5. Compare candidate vs baseline via strict gates:
- targeted recall delta
- FPR regression limit
- latency regression limit
6. Promote only if all gates pass; otherwise rollback from snapshot.

## Outputs
Per run under `evaluation/feedback_loop/<run_id>/`:
- `hard_examples_payload.jsonl`
- `hard_examples_url.jsonl`
- `candidate_dataset_manifest.json`
- `gating_report.json`
- `loop_summary.json`
- `baseline_models/` snapshot used for rollback and comparison

## Config
`config/feedback_loop.yaml` controls:
- gating thresholds
- dataset caps/ratios
- timeout and supported model list

## Training Script Integration
- `src/training/train_payload.py` now supports:
  - `--hard-examples-file <path>`
- `src/training/train_url.py` now supports:
  - `--hard-examples-file <path>`

## Operational Notes
- Loop is manual-trigger only by design.
- Scope is intentionally `payload,url` for V1 stability.
- Keep `--promote` off until reports are consistently healthy.
