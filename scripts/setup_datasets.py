#!/usr/bin/env python3
"""
Dataset Setup Script for AI-Hacking-Detection-ML

Downloads all required datasets for training and evaluation.
Run after cloning: python scripts/setup_datasets.py

Options:
    --all       Download all datasets (default)
    --minimal   Download only essential datasets (~50MB)
    --skip-large Skip datasets >100MB
"""
import argparse
import os
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download a file with progress indication."""
    print(f"  Downloading {desc or dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  ✓ Saved to {dest}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def run_cmd(cmd: str, cwd: Path = None) -> bool:
    """Run shell command."""
    try:
        subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                      capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Command failed: {e.stderr[:200] if e.stderr else str(e)}")
        return False


def setup_nsl_kdd() -> bool:
    """Download NSL-KDD network intrusion dataset (~25MB)."""
    print("\n[1/7] NSL-KDD Dataset")
    dest_dir = DATASETS_DIR / "network_intrusion" / "nsl_kdd"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        "KDDTrain+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        "KDDTest+.txt": "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
    }
    
    success = True
    for fname, url in files.items():
        dest = dest_dir / fname
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  ✓ {fname} already exists")
            continue
        if not download_file(url, dest, fname):
            success = False
    return success


def setup_urlhaus() -> bool:
    """Download URLhaus malicious URLs (~5MB)."""
    print("\n[2/7] URLhaus Malicious URLs")
    dest_dir = DATASETS_DIR / "url_analysis"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "urlhaus.csv"
    
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  ✓ urlhaus.csv already exists")
        return True
    
    return download_file(
        "https://urlhaus.abuse.ch/downloads/csv_recent/",
        dest, "URLhaus recent malicious URLs"
    )


def setup_tranco() -> bool:
    """Download Tranco top 1M domains (~10MB zip -> 22MB csv)."""
    print("\n[3/7] Tranco Top 1M Domains")
    dest_dir = DATASETS_DIR / "url_analysis"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    csv_dest = dest_dir / "top-1m.csv"
    if csv_dest.exists() and csv_dest.stat().st_size > 1000000:
        print(f"  ✓ top-1m.csv already exists")
        return True
    
    zip_dest = dest_dir / "tranco_top1m.csv.zip"
    if not download_file("https://tranco-list.eu/top-1m.csv.zip", zip_dest, "Tranco list"):
        return False
    
    try:
        with zipfile.ZipFile(zip_dest, 'r') as z:
            z.extractall(dest_dir)
        print(f"  ✓ Extracted to {csv_dest}")
        return True
    except Exception as e:
        print(f"  ✗ Extract failed: {e}")
        return False


def setup_kaggle_urls() -> bool:
    """Download Kaggle malicious URLs dataset (~10MB)."""
    print("\n[4/7] Kaggle Malicious URLs")
    dest_dir = DATASETS_DIR / "url_analysis"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "kaggle_malicious_urls.csv"
    
    if dest.exists() and dest.stat().st_size > 1000000:
        print(f"  ✓ kaggle_malicious_urls.csv already exists")
        return True
    
    return download_file(
        "https://raw.githubusercontent.com/incertum/cyber-matrix-ai/master/Malicious-URL-Detection-Deep-Learning/data/url_data_mega_deep_learning.csv",
        dest, "Kaggle malicious URLs"
    )


def setup_payloads_all_the_things() -> bool:
    """Clone PayloadsAllTheThings repository (~15MB)."""
    print("\n[5/7] PayloadsAllTheThings")
    dest_dir = DATASETS_DIR / "security_payloads" / "PayloadsAllTheThings"
    
    if dest_dir.exists() and (dest_dir / "README.md").exists():
        print(f"  ✓ PayloadsAllTheThings already exists")
        return True
    
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    print("  Cloning repository (shallow)...")
    return run_cmd(
        f"git clone --depth 1 https://github.com/swisskyrepo/PayloadsAllTheThings.git {dest_dir}"
    )


def setup_seclists() -> bool:
    """Clone SecLists Fuzzing folder only (~50MB)."""
    print("\n[6/7] SecLists (Fuzzing only)")
    dest_dir = DATASETS_DIR / "security_payloads" / "SecLists"
    
    if dest_dir.exists() and (dest_dir / "Fuzzing").exists():
        print(f"  ✓ SecLists/Fuzzing already exists")
        return True
    
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    print("  Cloning repository (sparse checkout)...")
    
    if not run_cmd(
        f"git clone --depth 1 --filter=blob:none --sparse https://github.com/danielmiessler/SecLists.git {dest_dir}"
    ):
        return False
    
    return run_cmd("git sparse-checkout set Fuzzing", cwd=dest_dir)


def setup_timeseries() -> bool:
    """Generate synthetic time-series data if missing (~57MB)."""
    print("\n[7/7] Time-Series Data")
    dest_dir = DATASETS_DIR / "timeseries"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    normal_file = dest_dir / "normal_traffic_improved.npy"
    attack_file = dest_dir / "attack_traffic_improved.npy"
    
    if normal_file.exists() and attack_file.exists():
        print(f"  ✓ Time-series data already exists")
        return True
    
    print("  Generating synthetic time-series data...")
    try:
        import numpy as np
        
        n_samples, seq_len, n_features = 10000, 100, 36
        
        # Normal traffic: smooth patterns
        normal = np.random.randn(n_samples, seq_len, n_features) * 0.3
        normal += np.sin(np.linspace(0, 4*np.pi, seq_len))[None, :, None] * 0.2
        
        # Attack traffic: anomalous spikes
        attack = np.random.randn(n_samples, seq_len, n_features) * 0.5
        spike_idx = np.random.randint(20, 80, n_samples)
        for i, idx in enumerate(spike_idx):
            attack[i, idx:idx+10, :] += np.random.randn(10, n_features) * 2
        
        np.save(normal_file, normal.astype(np.float32))
        np.save(attack_file, attack.astype(np.float32))
        print(f"  ✓ Generated {n_samples} samples each")
        return True
    except ImportError:
        print("  ✗ NumPy not installed. Run: pip install numpy")
        return False
    except Exception as e:
        print(f"  ✗ Generation failed: {e}")
        return False


def create_gitkeep_files():
    """Create .gitkeep files to preserve directory structure."""
    dirs = [
        "network_intrusion/nsl_kdd",
        "network_intrusion/cicids2017",
        "url_analysis",
        "security_payloads",
        "timeseries",
        "fraud_detection",
        "live_benign",
    ]
    for d in dirs:
        path = DATASETS_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        gitkeep = path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()


def setup_live_benign() -> bool:
    """Download live benign data for all models (~35-40GB)."""
    print("\n[8/8] Live Benign Data (for high precision/low FP)")
    
    script_path = BASE_DIR / "scripts" / "collect_live_data" / "download_all.py"
    if not script_path.exists():
        print(f"  ✗ download_all.py not found at {script_path}")
        return False
    
    print("  Running live benign data collector...")
    print("  This downloads ~35-40GB of benign samples for all models.")
    print("  Sources: Wikipedia, GitHub, Common Crawl, MAWI, etc.")
    
    try:
        # Run non-interactively with 'all' option
        result = subprocess.run(
            [sys.executable, str(script_path)],
            input="all\n",
            text=True,
            cwd=BASE_DIR,
            timeout=86400  # 24 hour timeout
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  ✗ Download timed out after 24 hours")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def count_dataset_samples() -> dict:
    """Count samples in all datasets."""
    counts = {}
    
    def count_lines(path: Path) -> int:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f)
        except:
            return 0
    
    def count_npy(path: Path) -> int:
        try:
            import numpy as np
            return np.load(path).shape[0]
        except:
            return 0
    
    # Count by category
    categories = {
        'benign_60m': DATASETS_DIR / 'benign_60m',
        'live_benign': DATASETS_DIR / 'live_benign',
        'security_payloads': DATASETS_DIR / 'security_payloads',
        'url_analysis': DATASETS_DIR / 'url_analysis',
        'network_intrusion': DATASETS_DIR / 'network_intrusion',
        'fraud_detection': DATASETS_DIR / 'fraud_detection',
        'host_behavior': DATASETS_DIR / 'host_behavior',
        'timeseries': DATASETS_DIR / 'timeseries',
    }
    
    for name, path in categories.items():
        if not path.exists():
            counts[name] = 0
            continue
        
        total = 0
        for f in path.rglob('*'):
            if f.is_file():
                if f.suffix in ('.jsonl', '.txt', '.csv'):
                    total += count_lines(f)
                elif f.suffix == '.npy':
                    total += count_npy(f)
        counts[name] = total
    
    return counts


def print_sample_summary():
    """Print summary of all dataset samples."""
    print("\n" + "=" * 60)
    print("DATASET SAMPLE COUNTS")
    print("=" * 60)
    
    counts = count_dataset_samples()
    
    total_benign = 0
    total_malicious = 0
    
    benign_cats = ['benign_60m', 'live_benign']
    malicious_cats = ['security_payloads']
    
    for name, count in counts.items():
        size_str = f"{count:,}" if count > 0 else "0"
        print(f"  {name}: {size_str} samples")
        
        if name in benign_cats:
            total_benign += count
        elif name in malicious_cats:
            total_malicious += count
    
    total = sum(counts.values())
    
    print(f"\n  {'─' * 40}")
    print(f"  TOTAL SAMPLES: {total:,}")
    print(f"  Estimated Benign: {total_benign:,}")
    print(f"  Estimated Malicious: {total_malicious:,}")
    
    if total > 0:
        benign_pct = total_benign / total * 100
        print(f"  Benign Ratio: {benign_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for AI-Hacking-Detection-ML")
    parser.add_argument("--minimal", action="store_true", help="Download only essential datasets")
    parser.add_argument("--skip-large", action="store_true", help="Skip datasets >100MB")
    parser.add_argument("--with-live-benign", action="store_true", help="Also download live benign data (~35-40GB)")
    parser.add_argument("--count-only", action="store_true", help="Only count existing samples, no downloads")
    args = parser.parse_args()
    
    print("=" * 60)
    print("AI-Hacking-Detection-ML Dataset Setup")
    print("=" * 60)
    
    if args.count_only:
        print_sample_summary()
        return 0
    
    os.chdir(BASE_DIR)
    create_gitkeep_files()
    
    results = {}
    
    # Essential datasets (always download)
    results["NSL-KDD"] = setup_nsl_kdd()
    results["URLhaus"] = setup_urlhaus()
    results["Tranco"] = setup_tranco()
    
    if not args.minimal:
        results["Kaggle URLs"] = setup_kaggle_urls()
        results["PayloadsAllTheThings"] = setup_payloads_all_the_things()
        
        if not args.skip_large:
            results["SecLists"] = setup_seclists()
            results["Time-Series"] = setup_timeseries()
    
    # Live benign data (optional, large)
    if args.with_live_benign:
        results["Live Benign Data"] = setup_live_benign()
    
    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    success = sum(results.values())
    total = len(results)
    
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    
    print(f"\nCompleted: {success}/{total} datasets")
    
    # Print sample counts
    print_sample_summary()
    
    if success == total:
        print("\n🎉 All datasets ready! You can now train models.")
    else:
        print("\n⚠️  Some downloads failed. Check errors above.")
        print("   You can re-run this script to retry failed downloads.")
    
    if not args.with_live_benign:
        print("\n💡 Tip: Run with --with-live-benign to download ~35-40GB of")
        print("   additional benign data for high precision / low FP training.")
    
    return 0 if success == total else 1


if __name__ == "__main__":
    sys.exit(main())
