"""Manual hard-example feedback loop for payload and URL models."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import yaml

# Allow running as `python src/feedback_loop/hard_example_loop.py` without importing `src.__init__`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feedback_loop.dataset_packager import write_manifest, write_model_dataset, write_summary
from feedback_loop.failure_ingest import ingest_failures
from feedback_loop.gating import GatingThresholds, evaluate_gates, evaluate_model_metrics
from feedback_loop.hard_example_generator import HardExampleGenerator
from feedback_loop.replay_buffer import build_replay_dataset, load_baseline_samples, load_previous_hard_examples
from feedback_loop.retrainer import restore_models, run_retraining, snapshot_models

SUPPORTED_MODELS = {"payload", "url"}


def _parse_models(value: str) -> List[str]:
    models = [m.strip() for m in value.split(",") if m.strip()]
    invalid = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid:
        raise ValueError(f"Unsupported models for feedback loop: {invalid}. Supported: {sorted(SUPPORTED_MODELS)}")
    return models


def _load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload if isinstance(payload, dict) else {}


def _records_to_eval_samples(records, model: str, cap: int = 4000) -> List[Dict]:
    out: List[Dict] = []
    for rec in records:
        if rec.model != model:
            continue
        out.append(
            {
                "text": rec.input_preview,
                "label": int(rec.expected),
                "category": rec.category,
            }
        )
        if len(out) >= cap:
            break
    return out


def run_loop(args) -> int:
    repo_root = Path(args.repo_root).resolve()
    config_path = repo_root / args.config
    config = _load_config(config_path)

    models = _parse_models(args.model)
    print(f"[feedback-loop] repo_root={repo_root}")
    print(f"[feedback-loop] models={','.join(models)} dry_run={bool(args.dry_run)} promote={bool(args.promote)}")
    print(f"[feedback-loop] input_dir={repo_root / args.input_dir} run_date={args.run_date or 'auto-latest'}")

    threshold_cfg = config.get("gating", {})
    thresholds = GatingThresholds(
        min_targeted_recall_delta=float(threshold_cfg.get("min_targeted_recall_delta", 0.02)),
        max_fpr_regression=float(threshold_cfg.get("max_fpr_regression", 0.005)),
        max_latency_regression_pct=float(threshold_cfg.get("max_latency_regression_pct", 0.10)),
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (repo_root / args.output_dir / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[feedback-loop] output_dir={output_dir}")

    ingest = ingest_failures(
        input_dir=repo_root / args.input_dir,
        models=models,
        run_date=args.run_date,
        max_failures_per_category=args.max_failures_per_category,
    )
    records = ingest["records"]
    print(f"[feedback-loop] source_files={json.dumps(ingest.get('source_files', {}), indent=2)}")
    print(f"[feedback-loop] ingested_records={len(records)}")

    if not records:
        summary = {
            "run_id": run_id,
            "status": "no_failures",
            "models": models,
            "source_files": ingest.get("source_files", {}),
            "message": "No failure records found for selected models.",
        }
        write_summary(output_dir, summary)
        print(f"[feedback-loop] no failures found. wrote summary: {output_dir / 'loop_summary.json'}")
        return 0

    generator = HardExampleGenerator(seed=args.seed, variants_per_failure=args.variants_per_failure)
    generated = generator.generate(records)
    print(f"[feedback-loop] generated_hard_examples={len(generated)}")

    model_outputs: Dict[str, Dict] = {}
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": models,
        "source_files": ingest.get("source_files", {}),
        "records_total": len(records),
        "generated_total": len(generated),
        "datasets": {},
    }

    for model in models:
        baseline = load_baseline_samples(repo_root, model=model, max_samples=args.baseline_max_samples)
        previous = load_previous_hard_examples(repo_root / args.output_dir, model=model, max_samples=args.previous_max_samples)
        replay = build_replay_dataset(
            model=model,
            baseline_samples=baseline,
            previous_hard_samples=previous,
            new_hard_samples=generated,
            replay_ratio=args.replay_ratio,
            hard_ratio_cap=args.hard_ratio_cap,
            seed=args.seed,
        )

        dataset_path = write_model_dataset(output_dir, model, replay["samples"])
        model_outputs[model] = {
            "dataset": str(dataset_path),
            "stats": replay["stats"],
        }
        manifest["datasets"][model] = model_outputs[model]
        print(
            f"[feedback-loop] model={model} replay_samples={replay['stats'].get('total_samples', len(replay['samples']))} "
            f"dataset={dataset_path}"
        )

    manifest_path = write_manifest(output_dir, manifest)
    print(f"[feedback-loop] manifest={manifest_path}")

    if args.dry_run:
        summary = {
            "run_id": run_id,
            "status": "dry_run_completed",
            "models": models,
            "manifest": str(manifest_path),
            "records_total": len(records),
            "generated_total": len(generated),
        }
        write_summary(output_dir, summary)
        print(f"[feedback-loop] dry-run complete. summary={output_dir / 'loop_summary.json'}")
        return 0

    # Snapshot current models before retraining
    snapshot_dir = output_dir / "baseline_models"
    snapshot_models(repo_root / args.models_dir, snapshot_dir, models)

    training_results = []
    for model in models:
        print(f"[feedback-loop] retraining model={model} ...")
        result = run_retraining(
            model=model,
            hard_examples_file=model_outputs[model]["dataset"],
            repo_root=repo_root,
            timeout_seconds=args.training_timeout_seconds,
        )
        training_results.append(result.to_dict())
        print(
            f"[feedback-loop] retraining model={model} status={result.status} "
            f"returncode={result.returncode} duration={result.duration_seconds:.1f}s"
        )
        if result.status != "completed":
            restore_models(repo_root / args.models_dir, snapshot_dir, models)
            summary = {
                "run_id": run_id,
                "status": "training_failed",
                "models": models,
                "training_results": training_results,
            }
            summary_path = write_summary(output_dir, summary)
            stderr_tail = (result.stderr or "")[-1200:]
            stdout_tail = (result.stdout or "")[-1200:]
            if stderr_tail:
                print("[feedback-loop] training stderr tail:")
                print(stderr_tail)
            elif stdout_tail:
                print("[feedback-loop] training stdout tail:")
                print(stdout_tail)
            print(f"[feedback-loop] training failed. summary={summary_path}")
            return 1

    # Strict gate evaluation
    gate_report = {
        "run_id": run_id,
        "thresholds": thresholds.__dict__,
        "models": {},
    }

    all_passed = True
    for model in models:
        eval_samples = _records_to_eval_samples(records, model=model)
        if not eval_samples:
            continue

        baseline_metrics = evaluate_model_metrics(snapshot_dir, model=model, eval_samples=eval_samples)
        candidate_metrics = evaluate_model_metrics(repo_root / args.models_dir, model=model, eval_samples=eval_samples)
        model_gate = evaluate_gates(baseline_metrics, candidate_metrics, thresholds)
        gate_report["models"][model] = model_gate
        all_passed = all_passed and model_gate["passed"]

    write_summary(output_dir, gate_report, filename="gating_report.json")
    print(f"[feedback-loop] gate_report={output_dir / 'gating_report.json'}")

    promoted = False
    if args.promote and all_passed:
        promoted = True
    else:
        restore_models(repo_root / args.models_dir, snapshot_dir, models)

    summary = {
        "run_id": run_id,
        "status": "completed",
        "models": models,
        "training_results": training_results,
        "gating_passed": all_passed,
        "promoted": promoted,
        "manifest": str(manifest_path),
    }
    write_summary(output_dir, summary)
    print(f"[feedback-loop] completed. summary={output_dir / 'loop_summary.json'} promoted={promoted}")

    return 0 if (not args.promote or promoted) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual hard-example feedback loop")
    parser.add_argument("--model", type=str, default="payload,url")
    parser.add_argument("--run-date", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default="evaluation/stress_test_v14")
    parser.add_argument("--output-dir", type=str, default="evaluation/feedback_loop")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--config", type=str, default="config/feedback_loop.yaml")
    parser.add_argument("--repo-root", type=str, default=".")
    parser.add_argument("--max-failures-per-category", type=int, default=5000)
    parser.add_argument("--variants-per-failure", type=int, default=3)
    parser.add_argument("--replay-ratio", type=float, default=0.6)
    parser.add_argument("--hard-ratio-cap", type=float, default=0.4)
    parser.add_argument("--baseline-max-samples", type=int, default=30000)
    parser.add_argument("--previous-max-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--training-timeout-seconds", type=int, default=7200)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run_loop(args)
    except Exception as exc:
        out = {
            "status": "error",
            "error": str(exc),
        }
        print(json.dumps(out, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
