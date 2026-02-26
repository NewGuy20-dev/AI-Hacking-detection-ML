#!/usr/bin/env python3
"""Build a balanced payload training dataset from benign/malicious sources."""
import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_guardrails import assert_allowed_training_paths


def _iter_text_records(paths: Iterable[Path]) -> List[str]:
    records: List[str] = []
    for path in paths:
        if path.is_dir():
            files = [p for p in path.rglob("*") if p.is_file()]
        elif path.exists():
            files = [path]
        else:
            continue

        for file_path in files:
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".jsonl":
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                        for line in handle:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except ValueError:
                                continue
                            text = str(
                                obj.get("text")
                                or obj.get("payload")
                                or obj.get("input")
                                or ""
                            ).strip()
                            if len(text) >= 3:
                                records.append(text)
                else:
                    text = file_path.read_text(encoding="utf-8", errors="ignore")
                    for line in text.splitlines():
                        line = line.strip()
                        if len(line) >= 3:
                            records.append(line)
            except OSError:
                continue
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare payload training dataset with target benign ratio.")
    parser.add_argument("--benign-ratio", type=float, default=0.50)
    parser.add_argument(
        "--benign-sources",
        nargs="+",
        default=["datasets/curated_benign", "datasets/live_benign"],
    )
    parser.add_argument(
        "--attack-sources",
        nargs="+",
        default=["datasets/security_payloads"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/security_payloads/payload_train_balanced.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-balance", type=lambda x: str(x).lower() == "true", default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    benign_sources = [Path(p) for p in args.benign_sources]
    attack_sources = [Path(p) for p in args.attack_sources]
    assert_allowed_training_paths(benign_sources + attack_sources, context="payload dataset source")

    benign = _iter_text_records(benign_sources)
    attack = _iter_text_records(attack_sources)
    benign = list(dict.fromkeys(benign))
    attack = list(dict.fromkeys(attack))

    if not benign or not attack:
        raise RuntimeError(f"Insufficient data. benign={len(benign)}, attack={len(attack)}")

    benign_ratio = max(0.05, min(0.95, args.benign_ratio))
    attack_ratio = 1.0 - benign_ratio
    total_cap = int(min(len(benign) / benign_ratio, len(attack) / attack_ratio))
    benign_target = int(total_cap * benign_ratio)
    attack_target = total_cap - benign_target

    random.shuffle(benign)
    random.shuffle(attack)
    benign_selected = benign[:benign_target]
    attack_selected = attack[:attack_target]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for text in benign_selected:
            f.write(json.dumps({"text": text, "label": 0}) + "\n")
        for text in attack_selected:
            f.write(json.dumps({"text": text, "label": 1}) + "\n")

    total = benign_target + attack_target
    actual_ratio = benign_target / max(total, 1)
    print(f"Wrote {total} samples to {out_path}")
    print(f"benign={benign_target}, attack={attack_target}, benign_ratio={actual_ratio:.4f}")
    if args.verify_balance and abs(actual_ratio - benign_ratio) > 0.01:
        raise RuntimeError(
            f"Balance check failed: target={benign_ratio:.4f}, actual={actual_ratio:.4f}"
        )


if __name__ == "__main__":
    main()
