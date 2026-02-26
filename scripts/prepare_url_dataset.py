#!/usr/bin/env python3
"""Prepare a balanced URL dataset with configurable benign ratio."""
import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable, List, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_guardrails import assert_allowed_training_paths


def _read_url_records(path: Path) -> List[str]:
    urls: List[str] = []
    if not path.exists():
        return urls
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                text = obj.get("url") or obj.get("text") or ""
                if text:
                    urls.append(str(text).strip())
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                candidate = row[0].strip()
                if candidate and "url" not in candidate.lower():
                    urls.append(candidate)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                text = line.strip()
                if text:
                    urls.append(text)
    return urls


def _collect(paths: Iterable[Path]) -> List[str]:
    records: List[str] = []
    for path in paths:
        records.extend(_read_url_records(path))
    return list(dict.fromkeys(records))


def _normalize_url(url: str) -> str:
    text = url.strip()
    if not text:
        return text
    if not text.startswith(("http://", "https://")):
        return f"http://{text}"
    return text


def _target_counts(benign_count: int, attack_count: int, benign_ratio: float) -> Tuple[int, int]:
    benign_ratio = max(0.05, min(0.95, benign_ratio))
    attack_ratio = 1.0 - benign_ratio
    total_cap = int(min(benign_count / benign_ratio, attack_count / attack_ratio))
    b = int(total_cap * benign_ratio)
    a = total_cap - b
    return b, a


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare URL dataset with benign/attack ratio control.")
    parser.add_argument("--benign-ratio", type=float, default=0.55)
    parser.add_argument("--tranco-supplement", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="datasets/url_analysis/url_train_balanced.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path("datasets/url_analysis")

    benign_sources = [
        root / "synthetic_benign_hard.txt",
        root / "url_benign_expansion.jsonl",
    ]
    if args.tranco_supplement:
        benign_sources.extend([root / "top-1m.csv", root / "domains" / "top-1m.csv"])

    attack_sources = [
        root / "synthetic_malicious_hard.txt",
        root / "url_malicious_expansion.jsonl",
        root / "real_malicious_urls.txt",
        root / "urlhaus.csv",
    ]
    assert_allowed_training_paths(benign_sources + attack_sources, context="url dataset source")

    benign = [_normalize_url(x) for x in _collect(benign_sources) if x]
    attack = [_normalize_url(x) for x in _collect(attack_sources) if x]

    if not benign or not attack:
        raise RuntimeError(f"Insufficient URL records. benign={len(benign)}, attack={len(attack)}")

    benign_target, attack_target = _target_counts(len(benign), len(attack), args.benign_ratio)
    random.shuffle(benign)
    random.shuffle(attack)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for url in benign[:benign_target]:
            f.write(json.dumps({"url": url, "label": 0}) + "\n")
        for url in attack[:attack_target]:
            f.write(json.dumps({"url": url, "label": 1}) + "\n")

    total = benign_target + attack_target
    print(f"Wrote {total} URLs to {out_path}")
    print(f"benign={benign_target}, attack={attack_target}, benign_ratio={benign_target / max(total,1):.4f}")


if __name__ == "__main__":
    main()
