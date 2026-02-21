# Threat Simulation Success Criteria

## Objective
Define pass/fail criteria for dynamic stress scenarios so model behavior can be evaluated against real-world threat pressure consistently.

## Required Metrics
- `TPR` (true positive rate): attack scenarios detected.
- `FPR` (false positive rate): benign scenarios incorrectly flagged.
- `P95 latency`: inference responsiveness under scenario load.
- `Drift sensitivity`: stability when category mix shifts over time.

## Baseline Gates
- `TPR >= 0.90` for high-risk classes (`sqli`, `xss`, `phishing`, `ddos`, `ransomware`, `zero_day`).
- `FPR <= 0.05` for benign and adversarial-benign buckets.
- `P95 latency <= 250ms` for API-path model predictions in stress runs.
- No sustained collapse in per-category accuracy over adaptive cycles.

## Replay Requirements
- Every stress run must use a logged `run_seed`.
- Every failed scenario must be written to model-specific `*_failures.jsonl`.
- Reproduction command format:
  - `python src/stress_test/stress_test_v14.py --model <model> --seed <run_seed> --duration <minutes>`

## Reporting
- Include per-model summary and per-category breakdown.
- Include top failing categories and representative failed scenario records.
- Include explicit residual risk statement for categories below threshold.
