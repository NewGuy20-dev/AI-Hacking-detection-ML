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


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


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
                        help='Estimated retrain time in hours for ETA printout (optional)')
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
    args = parser.parse_args()

    python = sys.executable
    today = date.today().isoformat()

    # ETA printout (rough)
    quick_minutes = 7 * args.quick_min  # 7 default models
    full_minutes = 7 * args.full_min
    fixed_hours = (quick_minutes + full_minutes) / 60.0 + 0.3
    if args.retrain_hours is None:
        print(f"\nEstimated total time: retrain + ~{fixed_hours:.1f} hours "
              f"(quick ~{quick_minutes}m + full ~{full_minutes}m)")
        print("Tip: pass --retrain-hours N to include retrain time in ETA.")
    else:
        eta_hours = args.retrain_hours + fixed_hours
        print(f"\nEstimated total time: ~{eta_hours:.1f} hours "
              f"(retrain ~{args.retrain_hours:.1f}h + quick ~{quick_minutes}m + full ~{full_minutes}m)")

    if not args.skip_handshake:
        print("Waiting for thermal guardian handshake...")
        if not handshake(Path(args.handshake_file), args.handshake_timeout):
            raise SystemExit("Thermal guardian handshake failed. Start thermal_guardian.py first.")

    if not args.skip_label_check:
        label_script = Path('scripts') / 'check_labels.py'
        if label_script.exists():
            run([python, str(label_script)])
        else:
            run([python, 'check_labels.py'])

    if not args.skip_train:
        run([python, 'scripts/train_rtx3050.py'])

    if not args.skip_profiles:
        run([
            python, 'scripts/build_stress_feature_profiles.py',
            '--max-samples', str(args.max_samples),
            '--max-lines', str(args.max_lines),
        ])

    if not args.skip_quick:
        run([
            python, 'src/stress_test/stress_test_v14.py',
            '--seed', str(args.seed),
            '--duration', str(args.quick_min),
        ])

    if not args.skip_analyze:
        run([
            python, 'scripts/analyze_stress_bias.py',
            '--date', today,
        ])

    if not args.skip_calibrate:
        run([
            python, 'scripts/calibrate_thresholds.py',
            '--date', today,
            '--fp-cap', str(args.fp_cap),
            '--recall-min', str(args.recall_min),
        ])

    if not args.skip_full:
        run([
            python, 'src/stress_test/stress_test_v14.py',
            '--seed', str(args.seed),
            '--duration', str(args.full_min),
        ])


if __name__ == '__main__':
    main()
