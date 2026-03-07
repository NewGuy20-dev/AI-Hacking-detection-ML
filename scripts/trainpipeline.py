#!/usr/bin/env python3
"""Run monthly retrain and validation sequence."""
import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path
import socket
import json
import time
import os
import warnings
from datetime import datetime, timezone


class TeeStream:
    """Mirror writes to terminal and log file."""

    def __init__(self, primary, mirror):
        self.primary = primary
        self.mirror = mirror

    def write(self, data):
        try:
            self.primary.write(data)
        except UnicodeEncodeError:
            safe = data.encode(getattr(self.primary, 'encoding', 'utf-8') or 'utf-8', errors='replace').decode(
                getattr(self.primary, 'encoding', 'utf-8') or 'utf-8', errors='replace'
            )
            self.primary.write(safe)
        self.mirror.write(data)
        self.mirror.flush()
        return len(data)

    def flush(self):
        self.primary.flush()
        self.mirror.flush()

    def isatty(self):
        return self.primary.isatty()


def run(cmd, env):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )
    assert process.stdout is not None

    decoder = __import__('codecs').getincrementaldecoder('utf-8')(errors='replace')
    line_buf = ''

    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            break
        text = decoder.decode(chunk)

        for ch in text:
            if ch == '\r':
                # Keep progress bars on one terminal line instead of expanding to many lines.
                if line_buf:
                    sys.stdout.write('\r' + line_buf)
                    sys.stdout.flush()
                    line_buf = ''
                else:
                    sys.stdout.write('\r')
                    sys.stdout.flush()
            elif ch == '\n':
                sys.stdout.write(line_buf + '\n')
                sys.stdout.flush()
                line_buf = ''
            else:
                line_buf += ch

    tail = decoder.decode(b'', final=True)
    if tail:
        line_buf += tail
    if line_buf:
        sys.stdout.write(line_buf + '\n')
        sys.stdout.flush()

    rc = process.wait()
    if rc != 0:
        raise SystemExit(f"Command failed with exit code {rc}: {' '.join(cmd)}")


DEFAULT_ETA_CONFIG = {
    "fallback_retrain_hours": 5.5,
    "history_window": 8,
    "ewma_alpha": 0.45,
    "min_runs_for_confidence": 3,
}

ETA_CONFIG_PATH = Path("config") / "training_eta.json"
ETA_HISTORY_PATH = Path("evaluation") / "metrics_logs" / "training_eta_history.json"
PIPELINE_LOG_DIR = Path("evaluation") / "metrics_logs" / "trainpipeline"


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_eta_config() -> dict:
    loaded = _load_json(ETA_CONFIG_PATH)
    if not isinstance(loaded, dict):
        return dict(DEFAULT_ETA_CONFIG)
    merged = dict(DEFAULT_ETA_CONFIG)
    merged.update(loaded)
    return merged


def load_eta_history() -> list:
    loaded = _load_json(ETA_HISTORY_PATH)
    if not isinstance(loaded, dict):
        return []
    runs = loaded.get("runs")
    if not isinstance(runs, list):
        return []
    return [r for r in runs if isinstance(r, dict)]


def save_eta_history(runs: list) -> None:
    ETA_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "runs": runs[-200:]}
    ETA_HISTORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def record_training_run(duration_seconds: float, retrain_forced: bool) -> None:
    runs = load_eta_history()
    runs.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "duration_hours": round(duration_seconds / 3600.0, 3),
            "retrain_forced": bool(retrain_forced),
            "status": "success",
        }
    )
    save_eta_history(runs)


def predict_retrain_hours(config: dict):
    runs = [
        r for r in load_eta_history()
        if r.get("status") == "success" and isinstance(r.get("duration_hours"), (int, float))
    ]
    if not runs:
        return float(config["fallback_retrain_hours"]), "fallback", 0

    window = int(config.get("history_window", 8))
    alpha = float(config.get("ewma_alpha", 0.45))
    recent = runs[-window:] if window > 0 else runs

    ewma = float(recent[0]["duration_hours"])
    for run in recent[1:]:
        ewma = alpha * float(run["duration_hours"]) + (1.0 - alpha) * ewma
    return ewma, "history_ewma", len(recent)


def handshake(handshake_file: Path, timeout_s: int):
    start = time.time()
    while time.time() - start < timeout_s:
        if handshake_file.exists():
            try:
                data = json.loads(handshake_file.read_text(encoding='utf-8'))
                token = data.get('token')
                port = data.get('port')
                if token and port:
                    try:
                        with socket.create_connection(('127.0.0.1', int(port)), timeout=5) as sock:
                            sock.sendall(f"HELLO {token}\n".encode('utf-8'))
                            resp = sock.recv(1024).decode('utf-8', errors='ignore').strip()
                            if resp == 'OK':
                                return True
                    except (OSError, ValueError):
                        pass
            except (json.JSONDecodeError, OSError, ValueError):
                pass
        time.sleep(1)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick-min', type=int, default=5)
    parser.add_argument('--full-min', type=int, default=45)
    parser.add_argument('--fp-cap', type=float, default=0.35)
    parser.add_argument('--recall-min', type=float, default=0.8)
    parser.add_argument('--max-samples', type=int, default=20000)
    parser.add_argument('--max-lines', type=int, default=200000)
    parser.add_argument('--retrain-hours', type=float, default=None,
                        help='Manual retrain estimate in hours (optional override)')
    parser.add_argument('--retrain', action='store_true',
                        help='Force full retrain (ignore existing models)')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-label-check', action='store_true')
    parser.add_argument('--skip-profiles', action='store_true')
    parser.add_argument('--skip-quick', action='store_true')
    parser.add_argument('--skip-analyze', action='store_true')
    parser.add_argument('--skip-calibrate', action='store_true')
    parser.add_argument('--skip-full', action='store_true')
    parser.add_argument('--skip-handshake', action='store_true')
    parser.add_argument('--handshake-timeout', type=int, default=30)
    parser.add_argument('--handshake-file', type=str,
                        default=str(Path('evaluation') / 'thermal_guardian' / 'handshake.json'))
    parser.add_argument('--log-file', type=str, default=None,
                        help='Optional path for live tee log file')
    args = parser.parse_args()

    python = sys.executable
    today = date.today().isoformat()
    eta_cfg = load_eta_config()
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # Ignore warnings in this process as well.
    warnings.filterwarnings("ignore")

    PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    default_log = PIPELINE_LOG_DIR / f"trainpipeline_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    log_path = Path(args.log_file) if args.log_file else default_log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_file:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(old_stdout, log_file)
        sys.stderr = TeeStream(old_stderr, log_file)
        try:
            print(f"Live log file: {log_path.resolve()}")

            # ETA printout (rough + predicted retrain)
            quick_minutes = 7 * args.quick_min  # 7 default models
            full_minutes = 7 * args.full_min
            fixed_hours = (quick_minutes + full_minutes) / 60.0 + 0.3
            if args.skip_train:
                print(f"\nEstimated total time: ~{fixed_hours:.1f} hours "
                      f"(train skipped, quick ~{quick_minutes}m + full ~{full_minutes}m)")
            elif args.retrain_hours is None:
                predicted_hours, source, sample_n = predict_retrain_hours(eta_cfg)
                eta_hours = predicted_hours + fixed_hours
                print(f"\nEstimated total time: ~{eta_hours:.1f} hours "
                      f"(retrain ~{predicted_hours:.1f}h + quick ~{quick_minutes}m + full ~{full_minutes}m)")
                if source == "history_ewma":
                    print(f"ETA source: training history ({sample_n} recent successful runs, EWMA).")
                else:
                    print("ETA source: fallback config (no successful training history yet).")
                print("Tip: pass --retrain-hours N to override auto prediction.")
            else:
                eta_hours = args.retrain_hours + fixed_hours
                print(f"\nEstimated total time: ~{eta_hours:.1f} hours "
                      f"(retrain ~{args.retrain_hours:.1f}h + quick ~{quick_minutes}m + full ~{full_minutes}m)")
                print("ETA source: manual override (--retrain-hours).")

            if not args.skip_handshake:
                print("Waiting for thermal guardian handshake...")
                if not handshake(Path(args.handshake_file), args.handshake_timeout):
                    raise SystemExit("Thermal guardian handshake failed. Start thermal_guardian.py first.")

            if not args.skip_label_check:
                label_script = Path('scripts') / 'check_labels.py'
                if label_script.exists():
                    run([python, str(label_script)], env)
                else:
                    run([python, 'check_labels.py'], env)

            if not args.skip_train:
                cmd = [python, 'scripts/train_rtx3050.py']
                if args.retrain:
                    cmd.append('--retrain')
                train_start = time.time()
                run(cmd, env)
                train_elapsed = time.time() - train_start
                record_training_run(train_elapsed, retrain_forced=args.retrain)
                print(f"Recorded training duration for ETA prediction: ~{train_elapsed / 3600.0:.2f}h")

            if not args.skip_profiles:
                run([
                    python, 'scripts/build_stress_feature_profiles.py',
                    '--max-samples', str(args.max_samples),
                    '--max-lines', str(args.max_lines),
                ], env)

            if not args.skip_quick:
                run([
                    python, 'src/stress_test/stress_test_v14.py',
                    '--seed', str(args.seed),
                    '--duration', str(args.quick_min),
                ], env)

            if not args.skip_analyze:
                run([
                    python, 'scripts/analyze_stress_bias.py',
                    '--date', today,
                ], env)

            if not args.skip_calibrate:
                run([
                    python, 'scripts/calibrate_thresholds.py',
                    '--date', today,
                    '--fp-cap', str(args.fp_cap),
                    '--recall-min', str(args.recall_min),
                ], env)

            if not args.skip_full:
                run([
                    python, 'src/stress_test/stress_test_v14.py',
                    '--seed', str(args.seed),
                    '--duration', str(args.full_min),
                ], env)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == '__main__':
    main()
