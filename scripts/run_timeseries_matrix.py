#!/usr/bin/env python3
"""Run the timeseries experiment matrix with unified logging."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SAFETY_RECALL = 0.2891418508851923


def _run_command(cmd: list[str], log_path: Path, passthrough: bool) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write(("COMMAND: " + " ".join(cmd) + "\n").encode("utf-8"))
        handle.flush()

    if passthrough:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return

    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            handle.write(chunk)
            handle.flush()
        exit_code = process.wait()
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, cmd)


def _write_weights(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def _is_collapsed(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = data.get("tuned_threshold_metrics") or {}
    if not metrics:
        return False
    tn = metrics.get("tn")
    fpr = metrics.get("fpr")
    if tn is None or fpr is None:
        return False
    return int(tn) == 0 or float(fpr) >= 0.999


def _copy_stress_artifacts(run_dir: Path, run_name: str, run_date: str) -> None:
    stress_dir = Path("evaluation") / "stress_test_v14" / run_date
    if not stress_dir.exists():
        return
    mapping = {
        f"timeseries_{run_date}.jsonl": f"{run_name}.stress.jsonl",
        f"timeseries_{run_date}_failures.jsonl": f"{run_name}.stress_failures.jsonl",
        f"timeseries_{run_date}_ops.json": f"{run_name}.stress_ops.json",
        f"run_manifest_{run_date}.json": f"{run_name}.stress_manifest.json",
        f"dashboard_{run_date}.html": f"{run_name}.stress_dashboard.html",
    }
    for src_name, dest_name in mapping.items():
        src = stress_dir / src_name
        if src.exists():
            shutil.copy2(src, run_dir / dest_name)


def _print_recall_delta(run_dir: Path, run_name: str) -> None:
    ops_path = run_dir / f"{run_name}.stress_ops.json"
    if not ops_path.exists():
        return
    data = json.loads(ops_path.read_text(encoding="utf-8"))
    recall = float(data.get("metrics", {}).get("recall", 0.0))
    delta = recall - SAFETY_RECALL
    relative = (delta / SAFETY_RECALL * 100.0) if SAFETY_RECALL else 0.0
    print(f"Recall delta vs safety: {delta:+.3f} ({relative:+.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run timeseries experiment matrix with unified logging.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=str, default="")
    parser.add_argument("--include-stress-probe", action="store_true")
    parser.add_argument("--include-stress-test", action="store_true")
    parser.add_argument("--stress-duration", type=int, default=5)
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help="Run child commands in passthrough mode to preserve tqdm progress bars.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir) if args.run_dir else (
        repo_root / "evaluation" / "experiments" / f"timeseries_{datetime.now():%Y%m%d_%H%M%S}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "matrix.log"

    focus_light_path = run_dir / "weights_focus_light.json"
    focus_medium_path = run_dir / "weights_focus_medium.json"
    focus_strong_path = run_dir / "weights_focus_strong.json"
    _write_weights(
        focus_light_path,
        {"ddos": 0.20, "portscan": 0.18, "exfiltration": 0.22, "c2": 0.22, "bruteforce": 0.18},
    )
    _write_weights(
        focus_medium_path,
        {"ddos": 0.15, "portscan": 0.15, "exfiltration": 0.25, "c2": 0.25, "bruteforce": 0.20},
    )
    _write_weights(
        focus_strong_path,
        {"ddos": 0.12, "portscan": 0.12, "exfiltration": 0.28, "c2": 0.28, "bruteforce": 0.20},
    )

    base_args = [
        sys.executable,
        "src/training/train_timeseries.py",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
    ]

    runs = [
        (
            "run1_baseline_run5",
            [
                "--stress-attack-count",
                "100000",
                "--attack-cap",
                "50000",
                "--stress-benign-count",
                "60000",
                "--stress-hard-negative-count",
                "40000",
                "--stress-val-count",
                "30000",
            ],
        ),
        (
            "run2_focus_light",
            [
                "--stress-attack-count",
                "100000",
                "--attack-cap",
                "50000",
                "--stress-benign-count",
                "60000",
                "--stress-hard-negative-count",
                "40000",
                "--stress-val-count",
                "30000",
                "--stress-attack-weights-file",
                str(focus_light_path),
                "--stress-val-weights-file",
                str(focus_light_path),
            ],
        ),
        (
            "run3_focus_medium",
            [
                "--stress-attack-count",
                "100000",
                "--attack-cap",
                "50000",
                "--stress-benign-count",
                "60000",
                "--stress-hard-negative-count",
                "40000",
                "--stress-val-count",
                "30000",
                "--stress-attack-weights-file",
                str(focus_medium_path),
                "--stress-val-weights-file",
                str(focus_medium_path),
            ],
        ),
        (
            "run4_focus_strong",
            [
                "--stress-attack-count",
                "100000",
                "--attack-cap",
                "50000",
                "--stress-benign-count",
                "60000",
                "--stress-hard-negative-count",
                "40000",
                "--stress-val-count",
                "30000",
                "--stress-attack-weights-file",
                str(focus_strong_path),
                "--stress-val-weights-file",
                str(focus_strong_path),
            ],
        ),
    ]

    for name, extra in runs:
        with log_path.open("ab") as handle:
            handle.write(f"===== {name} : TRAIN =====\n".encode("utf-8"))
        _run_command(base_args + extra, log_path, args.passthrough)

        thresholds = repo_root / "config" / "model_thresholds.yaml"
        manifest = repo_root / "models" / "timeseries_lstm_training_manifest.json"
        if thresholds.exists():
            shutil.copy2(thresholds, run_dir / f"{name}.thresholds.yaml")
        if manifest.exists():
            manifest_snapshot = run_dir / f"{name}.manifest.json"
            shutil.copy2(manifest, manifest_snapshot)
        else:
            manifest_snapshot = run_dir / f"{name}.manifest.json"

        if args.include_stress_probe:
            with log_path.open("ab") as handle:
                handle.write(f"===== {name} : DIAGNOSE =====\n".encode("utf-8"))
            _run_command(
                [
                    sys.executable,
                    "scripts/diagnose_timeseries_artifact.py",
                    "--stress-sample-count",
                    "200",
                ],
                log_path,
                args.passthrough,
            )

        if args.include_stress_test:
            if _is_collapsed(manifest_snapshot):
                with log_path.open("ab") as handle:
                    handle.write(b"Detected collapse in manifest metrics; skipping stress test.\n")
                continue
            with log_path.open("ab") as handle:
                handle.write(f"===== {name} : STRESS_TEST =====\n".encode("utf-8"))
            stress_failed = False
            try:
                _run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.stress_test.stress_test_v14",
                        "--model",
                        "timeseries",
                        "--seed",
                        str(args.seed),
                        "--duration",
                        str(args.stress_duration),
                    ],
                    log_path,
                    args.passthrough,
                )
            except subprocess.CalledProcessError:
                stress_failed = True
                with log_path.open("ab") as handle:
                    handle.write(b"Stress test exited non-zero; artifacts will be copied if available.\n")
            _copy_stress_artifacts(run_dir, name, datetime.now().strftime("%Y-%m-%d"))
            _print_recall_delta(run_dir, name)
            if stress_failed:
                print("Stress test exited non-zero; artifacts copied if available.")

    print(f"All runs completed. Logs and snapshots saved to {run_dir}")
    print(f"Combined log: {log_path}")


if __name__ == "__main__":
    main()
