"""Replay buffer construction for feedback-loop retraining."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List


def _read_lines(path: Path, max_lines: int = 20000) -> List[str]:
    out: List[str] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                break
            line = line.strip()
            if line:
                out.append(line)
    return out


def load_baseline_samples(base_dir: str | Path, model: str, max_samples: int = 30000) -> List[Dict]:
    """Load baseline replay samples for model from existing dataset folders."""
    base = Path(base_dir)
    rng = random.Random(42)
    samples: List[Dict] = []

    if model == "payload":
        mal_dir = base / "datasets" / "security_payloads"
        ben_dir = base / "datasets" / "curated_benign"

        # Malicious payload seeds
        for folder in ["injection", "fuzzing", "misc"]:
            for file in (mal_dir / folder).rglob("*"):
                if not file.is_file() or file.suffix not in ("", ".txt", ".lst", ".list"):
                    continue
                for line in _read_lines(file, max_lines=1000):
                    if len(line) > 3:
                        samples.append({"model": "payload", "text": line, "label": 1, "origin": "baseline"})

        # Benign payload seeds
        for file in ben_dir.rglob("*.txt"):
            for line in _read_lines(file, max_lines=3000):
                if len(line) > 2:
                    samples.append({"model": "payload", "text": line, "label": 0, "origin": "baseline"})

    elif model == "url":
        url_dir = base / "datasets" / "url_analysis"
        kaggle = url_dir / "kaggle_malicious_urls.csv"
        if kaggle.exists():
            for line in _read_lines(kaggle, max_lines=max_samples):
                if line.lower().startswith("url"):
                    continue
                parts = line.rsplit(",", 1)
                if len(parts) != 2:
                    continue
                url, label = parts[0].strip(), parts[1].strip()
                if not url:
                    continue
                if not url.startswith(("http://", "https://")):
                    url = f"http://{url}"
                if label in {"0", "1"}:
                    samples.append({"model": "url", "text": url, "label": int(label), "origin": "baseline"})

        tranco = url_dir / "top-1m.csv"
        if tranco.exists():
            for line in _read_lines(tranco, max_lines=max_samples):
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                domain = parts[1].strip()
                if domain:
                    samples.append({"model": "url", "text": f"https://{domain}/", "label": 0, "origin": "baseline"})

    rng.shuffle(samples)
    return samples[:max_samples]


def load_previous_hard_examples(history_dir: str | Path, model: str, max_samples: int = 20000) -> List[Dict]:
    """Load historical hard examples from previous loop outputs."""
    root = Path(history_dir)
    if not root.exists():
        return []

    out: List[Dict] = []
    files = sorted(root.glob("**/hard_examples_*.jsonl"))
    for file in files:
        with open(file, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("model") != model:
                    continue
                if "text" not in item or "label" not in item:
                    continue
                out.append(item)
                if len(out) >= max_samples:
                    return out
    return out


def _dedupe_samples(samples: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for sample in samples:
        key = (
            str(sample.get("model", "")),
            int(sample.get("label", 0)),
            str(sample.get("text", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(sample)
    return out


def build_replay_dataset(
    model: str,
    baseline_samples: List[Dict],
    previous_hard_samples: List[Dict],
    new_hard_samples: List[Dict],
    replay_ratio: float = 0.6,
    hard_ratio_cap: float = 0.4,
    seed: int = 42,
) -> Dict:
    """Build final retraining set with replay and hard-example controls."""
    rng = random.Random(seed)
    baseline = _dedupe_samples([s for s in baseline_samples if s.get("model") == model])
    hard = _dedupe_samples(
        [s for s in previous_hard_samples if s.get("model") == model]
        + [s for s in new_hard_samples if s.get("model") == model]
    )

    if not baseline and not hard:
        return {"samples": [], "stats": {"total": 0, "baseline": 0, "hard": 0}}

    base_target = max(1, int(len(baseline) * replay_ratio)) if baseline else 0
    hard_target = min(len(hard), max(1, int((base_target + len(hard)) * hard_ratio_cap))) if hard else 0

    selected_baseline = baseline[:base_target]
    selected_hard = hard[:hard_target]

    # Ensure every new hard sample has a chance to land in dataset
    if new_hard_samples:
        for sample in new_hard_samples:
            if sample.get("model") == model and sample not in selected_hard:
                selected_hard.append(sample)

    final_samples = _dedupe_samples(selected_baseline + selected_hard)

    # Class rebalance soft pass
    pos = [s for s in final_samples if int(s.get("label", 0)) == 1]
    neg = [s for s in final_samples if int(s.get("label", 0)) == 0]
    if pos and neg:
        keep = min(len(pos), len(neg))
        rng.shuffle(pos)
        rng.shuffle(neg)
        final_samples = pos[:keep] + neg[:keep]

    rng.shuffle(final_samples)

    return {
        "samples": final_samples,
        "stats": {
            "model": model,
            "total": len(final_samples),
            "baseline_pool": len(baseline),
            "hard_pool": len(hard),
            "selected_baseline": len(selected_baseline),
            "selected_hard": len(selected_hard),
            "positive": sum(1 for s in final_samples if int(s.get("label", 0)) == 1),
            "negative": sum(1 for s in final_samples if int(s.get("label", 0)) == 0),
        },
    }
