#!/usr/bin/env python3
"""Create stratified holdout test set for final validation."""
import json
import random
from pathlib import Path


def create_holdout_test_set(base_path: Path, holdout_ratio: float = 0.1):
    """Create stratified holdout test set."""
    holdout_dir = base_path / "datasets" / "holdout_test"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating holdout test set...")
    
    # Collect malicious samples
    malicious = []
    payloads_dir = base_path / "datasets" / "security_payloads"
    for folder in ["injection", "fuzzing", "misc"]:
        folder_path = payloads_dir / folder
        if folder_path.exists():
            for f in folder_path.rglob("*"):
                try:
                    if not f.is_file() or f.suffix not in ("", ".txt", ".lst"):
                        continue
                    for line in f.read_text(errors="ignore").splitlines()[:200]:
                        if line.strip() and len(line.strip()) > 3:
                            malicious.append({"text": line.strip(), "label": 1, "source": str(f.name)})
                except (OSError, PermissionError, IOError):
                    continue
    
    # Collect benign samples
    benign = []
    
    # From curated benign
    benign_dir = base_path / "datasets" / "curated_benign"
    if benign_dir.exists():
        for f in benign_dir.glob("*.txt"):
            try:
                for line in f.read_text(errors="ignore").splitlines()[:500]:
                    if line.strip():
                        benign.append({"text": line.strip(), "label": 0, "source": f.name})
            except: pass
    
    # From FP test set
    fp_test = base_path / "datasets" / "fp_test_500k.jsonl"
    if fp_test.exists():
        try:
            with open(fp_test, "r") as f:
                for i, line in enumerate(f):
                    if i >= 5000: break
                    obj = json.loads(line)
                    if obj.get("text"):
                        benign.append({"text": obj["text"], "label": 0, "source": "fp_test"})
        except: pass
    
    # Shuffle and split
    random.shuffle(malicious)
    random.shuffle(benign)
    
    # Take holdout portion
    mal_holdout = malicious[:int(len(malicious) * holdout_ratio)]
    ben_holdout = benign[:int(len(benign) * holdout_ratio)]
    
    # Balance classes
    min_count = min(len(mal_holdout), len(ben_holdout))
    mal_holdout = mal_holdout[:min_count]
    ben_holdout = ben_holdout[:min_count]
    
    # Combine and shuffle
    holdout = mal_holdout + ben_holdout
    random.shuffle(holdout)
    
    # Save holdout set
    holdout_file = holdout_dir / "holdout_test.jsonl"
    with open(holdout_file, "w") as f:
        for item in holdout:
            f.write(json.dumps(item) + "\n")
    
    # Save metadata
    meta = {
        "total_samples": len(holdout),
        "malicious_count": len(mal_holdout),
        "benign_count": len(ben_holdout),
        "holdout_ratio": holdout_ratio,
        "sources": {
            "malicious": list(set(m["source"] for m in mal_holdout)),
            "benign": list(set(b["source"] for b in ben_holdout))[:10],
        }
    }
    
    with open(holdout_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Holdout set: {len(holdout):,} samples")
    print(f"  Malicious: {len(mal_holdout):,}")
    print(f"  Benign: {len(ben_holdout):,}")
    print(f"  Saved to: {holdout_file}")
    
    return holdout_file


if __name__ == "__main__":
    base = Path(__file__).parent.parent
    create_holdout_test_set(base)
