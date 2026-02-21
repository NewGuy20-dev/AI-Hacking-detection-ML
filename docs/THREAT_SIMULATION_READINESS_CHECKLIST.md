# Threat Simulation Readiness Checklist

## Scope
- Dynamic scenario generators for: `payload`, `url`, `timeseries`, `meta`, `fraud`, `host`, `network`, `anomaly`.
- Validation source: `evaluation/stress_test_v14/threat_sim_readiness_report.md`.

## Checklist
- [x] Representative dynamic scenarios executed across all model scenario generators.
- [x] Malicious category coverage confirmed for all configured threat classes.
- [x] Seed replayability confirmed for deterministic scenario reproduction.
- [x] Failing-corpus logging path added (`*_failures.jsonl`) for forensic replay.
- [x] Threat simulation criteria documented (`docs/THREAT_SIMULATION_CRITERIA.md`).

## Residual Risks
- Full inference-path stress execution still depends on local ML runtime dependencies (`torch`, `joblib`, `scikit-learn`) being installed.
- Dynamic generator realism is materially improved, but production parity still depends on model retraining cadence and live-data refresh quality.
- Thresholds in criteria are defined; enforcement should be automated in CI/CD gating once runtime dependencies are fully available in the target runner.

## Replay Command
- `python src/stress_test/stress_test_v14.py --model <model> --seed <seed> --duration <minutes>`
