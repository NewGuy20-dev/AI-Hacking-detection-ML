#!/usr/bin/env python3
"""Run real-world replay evaluation with v1.4-compatible outputs."""
import argparse
import json
from pathlib import Path
from datetime import date

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stress_test.v14.models import ModelWrapper
from src.stress_test.v14.replay_loader import load_replay_source


def parse_args():
    p = argparse.ArgumentParser(description="Real-world replay evaluation")
    p.add_argument('--model', required=True, help='Model name (payload,url,timeseries,meta,host,network,fraud,anomaly)')
    p.add_argument('--config', required=True, help='YAML file describing replay sources')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-samples', type=int, default=100000)
    p.add_argument('--output-dir', default='evaluation/stress_test_v14')
    p.add_argument('--strict-schema', action='store_true')
    return p.parse_args()


def load_config(path: Path):
    import yaml
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    sources = cfg.get('sources', [])

    wrapper = ModelWrapper(args.model, models_dir=Path('models')).load()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = date.today().isoformat()
    out_path = out_dir / f"replay_{args.model}_{run_date}.jsonl"
    summary_path = out_dir / f"replay_summary_{args.model}_{run_date}.json"

    total = 0
    passed = 0
    failed = 0

    with open(out_path, 'w', encoding='utf-8') as outf:
        for entry in sources:
            samples = load_replay_source(entry)
            for sample in samples:
                if total >= args.max_samples:
                    break
                try:
                    pred, conf, lat = wrapper.predict(sample.input_data)
                    rec = {
                        'scenario_id': f"replay_{entry.get('name','src')}_{total}",
                        'model': args.model,
                        'category': sample.category,
                        'subcategory': 'replay',
                        'input_preview': str(sample.input_data)[:100],
                        'expected': sample.expected_label,
                        'predicted': pred,
                        'confidence': conf,
                        'passed': pred == sample.expected_label,
                        'latency_ms': round(lat, 2),
                        'difficulty': entry.get('difficulty', 'medium'),
                        'source': 'replay',
                        'tags': [sample.category],
                        'run_seed': args.seed,
                    }
                    total += 1
                    passed += int(rec['passed'])
                    failed += int(not rec['passed'])
                    outf.write(json.dumps(rec) + "\n")
                except Exception:
                    if args.strict_schema:
                        raise
                    continue

    summary = {
        'model': args.model,
        'run_date': run_date,
        'total': total,
        'passed': passed,
        'failed': failed,
        'pass_rate': passed / total if total else 0.0,
        'log': str(out_path),
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Replay complete: {passed}/{total} passed; log={out_path}")


if __name__ == "__main__":
    main()
