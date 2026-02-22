#!/usr/bin/env python3
"""
Thermal Guardian - Kill training if GPU temp >= 90°C.
User resumes manually with --resume flag after cooldown.

Usage:
    python scripts/thermal_guardian.py              # Default 90°C threshold
    python scripts/thermal_guardian.py --threshold 85
"""
import subprocess
import signal
import sys
import time
import os
import argparse
from datetime import datetime
from pathlib import Path
import threading
import socket
import json
import secrets

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
DEFAULT_HANDSHAKE_FILE = Path('evaluation') / 'thermal_guardian' / 'handshake.json'


def get_gpu_temp():
    """Get GPU temperature. Returns -1 on failure."""
    # Try pynvml first (faster)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except:
        pass
    
    # Fallback to nvidia-smi
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            return int(r.stdout.strip().split('\n')[0])
    except:
        pass
    return -1


def find_training_pid():
    """Find PID of training process."""
    try:
        # Try pgrep first (Linux/WSL)
        r = subprocess.run(['pgrep', '-f', 'train_rtx3050'], capture_output=True, text=True)
        if r.stdout.strip():
            return int(r.stdout.strip().split('\n')[0])
    except:
        pass
    
    # Fallback: check tasklist on Windows
    try:
        r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                          capture_output=True, text=True)
        # Would need to parse and match - simplified for now
    except:
        pass
    return None


def notify_discord(title, message, color=0xe74c3c):
    """Send Discord notification."""
    if not HAS_REQUESTS or not DISCORD_WEBHOOK:
        return
    try:
        requests.post(DISCORD_WEBHOOK, json={
            "embeds": [{
                "title": title,
                "description": message,
                "color": color,
                "timestamp": datetime.utcnow().isoformat()
            }]
        }, timeout=10)
    except:
        pass


def log(msg):
    """Log with timestamp."""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}")


def _handshake_server(bind_addr: str, token: str, ready_evt: threading.Event,
                      stop_evt: threading.Event, handshake_file: Path, error_ref: dict):
    """Start a one-shot TCP handshake server."""
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((bind_addr, 0))
        sock.listen(1)
        port = sock.getsockname()[1]
    except Exception as exc:
        error_ref['error'] = str(exc)
        ready_evt.set()
        if sock:
            try:
                sock.close()
            except Exception:
                pass
        return

    handshake_file.parent.mkdir(parents=True, exist_ok=True)
    with handshake_file.open('w', encoding='utf-8') as f:
        json.dump({
            'token': token,
            'port': port,
            'created_at': datetime.now().isoformat(),
        }, f, indent=2)

    log(f"Handshake server listening on {bind_addr}:{port}")
    ready_evt.set()

    sock.settimeout(1.0)
    try:
        while not stop_evt.is_set():
            try:
                conn, _addr = sock.accept()
            except socket.timeout:
                continue
            with conn:
                try:
                    data = conn.recv(1024).decode('utf-8', errors='ignore').strip()
                except Exception:
                    data = ''
                if data == f"HELLO {token}":
                    conn.sendall(b"OK\n")
                    stop_evt.set()
                else:
                    conn.sendall(b"ERR\n")
            if stop_evt.is_set():
                break
    except Exception as exc:
        error_ref['error'] = str(exc)
    finally:
        if sock:
            sock.close()
        if handshake_file.exists():
            try:
                handshake_file.unlink()
                log("Handshake token file deleted")
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description='GPU Thermal Guardian')
    parser.add_argument('--threshold', type=int, default=90, help='Kill threshold (default: 90°C)')
    parser.add_argument('--interval', type=float, default=5.0, help='Poll interval (default: 5s)')
    handshake_group = parser.add_mutually_exclusive_group()
    handshake_group.add_argument('--handshake', dest='handshake', action='store_true', help='Enable handshake server')
    handshake_group.add_argument('--no-handshake', dest='handshake', action='store_false', help='Disable handshake server')
    parser.set_defaults(handshake=True)
    parser.add_argument('--handshake-timeout', type=int, default=180, help='Handshake server max lifetime (seconds)')
    parser.add_argument('--handshake-file', type=str, default=str(DEFAULT_HANDSHAKE_FILE),
                        help='Handshake token file path')
    parser.add_argument('--handshake-bind', type=str, default='127.0.0.1', help='Bind address for handshake server')
    args = parser.parse_args()
    
    threshold = args.threshold
    interval = args.interval
    
    log(f"🛡️ Thermal Guardian started")
    log(f"   Threshold: {threshold}°C")
    log(f"   Poll interval: {interval}s")
    log(f"   Will kill training and exit if temp >= {threshold}°C")
    if HAS_REQUESTS and not DISCORD_WEBHOOK:
        log("ERROR: DISCORD_WEBHOOK not set; Discord alerts disabled")

    handshake_file = Path(args.handshake_file)
    if not args.handshake:
        log("Handshake server disabled by --no-handshake")
    else:
        token = secrets.token_hex(16)
        ready_evt = threading.Event()
        stop_evt = threading.Event()
        error_ref = {}
        server_thread = threading.Thread(
            target=_handshake_server,
            args=(args.handshake_bind, token, ready_evt, stop_evt, handshake_file, error_ref),
            daemon=True
        )
        server_thread.start()
        ready = ready_evt.wait(timeout=5)
        if not ready:
            log("ERROR: Handshake server failed to start (timeout)")
            stop_evt.set()
            server_thread.join(timeout=2)
            raise SystemExit("Handshake server failed to start")
        if error_ref.get('error'):
            log(f"ERROR: Handshake server failed to start: {error_ref['error']}")
            stop_evt.set()
            server_thread.join(timeout=2)
            raise SystemExit("Handshake server failed to start")
        log("Handshake server ready")

        def _handshake_timeout():
            time.sleep(args.handshake_timeout)
            if not stop_evt.is_set():
                log("Handshake timeout reached; stopping handshake server")
                stop_evt.set()
                if handshake_file.exists():
                    try:
                        handshake_file.unlink()
                        log("Handshake token file deleted")
                    except Exception:
                        pass

        timeout_thread = threading.Thread(target=_handshake_timeout, daemon=True)
        timeout_thread.start()
    
    last_log = 0
    
    while True:
        temp = get_gpu_temp()
        
        if temp < 0:
            log("⚠️ Cannot read GPU temperature")
            time.sleep(interval * 2)
            continue
        
        # Log every 60 seconds
        now = time.time()
        if now - last_log >= 60:
            log(f"GPU: {temp}°C")
            last_log = now
        
        # Check threshold
        if temp >= threshold:
            log(f"🔥 GPU at {temp}°C >= {threshold}°C - STOPPING TRAINING")
            
            pid = find_training_pid()
            if pid:
                log(f"Sending SIGTERM to training process (PID {pid})")
                try:
                    os.kill(pid, signal.SIGTERM)
                    log("SIGTERM sent - training will save checkpoint and exit")
                except Exception as e:
                    log(f"Failed to kill process: {e}")
            else:
                log("No training process found")
            
            notify_discord(
                "🔥 Training Stopped - Thermal Protection",
                f"GPU temperature: **{temp}°C**\nThreshold: {threshold}°C\n\nResume with: `python scripts/train_rtx3050.py --resume`",
                0xe74c3c
            )
            
            log("=" * 50)
            log("Training stopped. To resume after cooldown:")
            log("  python scripts/train_rtx3050.py --model url --resume")
            log("=" * 50)
            sys.exit(0)
        
        time.sleep(interval)


if __name__ == '__main__':
    main()
