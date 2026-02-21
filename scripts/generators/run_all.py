#!/usr/bin/env python3
"""Parallel job orchestrator for synthetic data generation (max 2 concurrent)."""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Job:
    name: str
    script: str
    total: int
    output: str
    priority: int  # Lower = higher priority

JOBS = [
    Job("Payload", "gen_payload.py", 15_000_000, "datasets/benign_60m|datasets/security_payloads", 1),
    Job("URL", "gen_url.py", 10_000_000, "datasets/url_analysis", 1),
    Job("Network", "gen_network.py", 10_000_000, "datasets/network_intrusion", 2),
    Job("Host", "gen_host.py", 10_000_000, "datasets/host_behavior", 2),
    Job("Timeseries", "gen_timeseries.py", 5_000_000, "datasets/timeseries", 3),
    Job("Fraud", "gen_fraud.py", 5_000_000, "datasets/fraud_detection", 3),
]

def run_job(job: Job, python_exe: str, base_dir: Path) -> tuple[str, bool, float]:
    """Run a single generator job."""
    script_path = base_dir / "scripts" / "generators" / job.script
    start = time.time()
    
    try:
        print(f"\n[{job.name}] Starting...")
        # Handle payload special case with two output directories
        if job.name == "Payload":
            benign_dir, malicious_dir = job.output.split("|")
            result = subprocess.run(
                [python_exe, str(script_path), "--total", str(job.total), 
                 "--benign-output", str(base_dir / benign_dir),
                 "--malicious-output", str(base_dir / malicious_dir)],
                cwd=str(base_dir)
            )
        else:
            result = subprocess.run(
                [python_exe, str(script_path), "--total", str(job.total), "--output", str(base_dir / job.output)],
                cwd=str(base_dir)
            )
        success = result.returncode == 0
        if not success:
            print(f"\n[{job.name}] FAILED")
        return job.name, success, time.time() - start
    except Exception as e:
        print(f"\n[{job.name}] ERROR: {e}")
        return job.name, False, time.time() - start

def main():
    parser = argparse.ArgumentParser(description="Run synthetic data generators in parallel")
    parser.add_argument("--max-workers", type=int, default=2, help="Max concurrent jobs")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable")
    parser.add_argument("--dry-run", action="store_true", help="Show job queue without running")
    parser.add_argument("--jobs", nargs="+", choices=[j.name.lower() for j in JOBS], help="Run specific jobs only")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent.parent.parent
    
    # Filter jobs if specified
    jobs = JOBS
    if args.jobs:
        jobs = [j for j in JOBS if j.name.lower() in args.jobs]
    
    # Sort by priority
    jobs = sorted(jobs, key=lambda j: j.priority)
    
    total_samples = sum(j.total for j in jobs)
    print(f"=== Synthetic Data Generation ===")
    print(f"Total: {total_samples:,} samples across {len(jobs)} jobs")
    print(f"Max concurrent: {args.max_workers}")
    print(f"\nJob Queue:")
    for i, job in enumerate(jobs, 1):
        print(f"  {i}. {job.name}: {job.total:,} samples -> {job.output}")
    
    if args.dry_run:
        print("\n[DRY RUN] No jobs executed.")
        return
    
    print(f"\nStarting generation...\n")
    start_time = time.time()
    results = {}
    
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_job, job, args.python, base_dir): job for job in jobs}
        
        for future in as_completed(futures):
            job = futures[future]
            name, success, duration = future.result()
            status = "✓" if success else "✗"
            results[name] = (success, duration)
            print(f"[{status}] {name} completed in {duration/60:.1f} min")
    
    # Summary
    total_time = time.time() - start_time
    successful = sum(1 for s, _ in results.values() if s)
    print(f"\n=== Summary ===")
    print(f"Completed: {successful}/{len(jobs)} jobs")
    print(f"Total time: {total_time/60:.1f} min")
    
    for name, (success, duration) in results.items():
        status = "✓" if success else "✗"
        print(f"  [{status}] {name}: {duration/60:.1f} min")

if __name__ == "__main__":
    main()
