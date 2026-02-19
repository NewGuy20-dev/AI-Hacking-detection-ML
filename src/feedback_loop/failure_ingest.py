"""Failure ingestion for hard-example feedback loop."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class FailureRecord:
    """Normalized failure sample record from stress-test logs."""

    scenario_id: str
    model: str
    category: str
    subcategory: str
    expected: int
    predicted: int
    confidence: float
    difficulty: str
    source: str
    tags: List[str]
    run_seed: Optional[int]
    input_preview: str
    timestamp: str
    error: Optional[str]
    record_hash: str

    def to_dict(self) -> Dict:
        return asdict(self)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _hash_record(model: str, category: str, difficulty: str, input_preview: str) -> str:
    key = f"{model}|{category}|{difficulty}|{_normalize_text(input_preview)}"
    return hashlib.sha256(key.encode("utf-8", errors="ignore")).hexdigest()


def _extract_date_token(path: Path) -> str:
    stem = path.stem
    # payload_2026-01-17_failures -> 2026-01-17
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1] == "failures":
        return parts[-2]
    return ""


def _select_failure_file(input_dir: Path, model: str, run_date: Optional[str]) -> Optional[Path]:
    if run_date:
        candidate = input_dir / f"{model}_{run_date}_failures.jsonl"
        return candidate if candidate.exists() else None

    matches = sorted(input_dir.glob(f"{model}_*_failures.jsonl"))
    if not matches:
        return None
    matches.sort(key=lambda p: _extract_date_token(p))
    return matches[-1]


def _iter_records(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield payload


def ingest_failures(
    input_dir: str | Path,
    models: List[str],
    run_date: Optional[str] = None,
    max_failures_per_category: int = 5000,
) -> Dict:
    """Load and normalize failure records for selected models."""
    input_dir = Path(input_dir)
    seen_hashes = set()
    records: List[FailureRecord] = []
    per_model_files: Dict[str, Optional[str]] = {}

    for model in models:
        source_file = _select_failure_file(input_dir, model, run_date)
        per_model_files[model] = str(source_file) if source_file else None
        if not source_file:
            continue

        category_counts: Dict[str, int] = {}
        for raw in _iter_records(source_file):
            category = str(raw.get("category", "unknown"))
            if category_counts.get(category, 0) >= max_failures_per_category:
                continue

            preview = str(raw.get("input_preview", ""))
            rec_hash = _hash_record(
                model=str(raw.get("model", model)),
                category=category,
                difficulty=str(raw.get("difficulty", "medium")),
                input_preview=preview,
            )
            if rec_hash in seen_hashes:
                continue
            seen_hashes.add(rec_hash)

            tags = raw.get("tags")
            if not isinstance(tags, list):
                tags = [category]

            record = FailureRecord(
                scenario_id=str(raw.get("scenario_id", "")),
                model=str(raw.get("model", model)),
                category=category,
                subcategory=str(raw.get("subcategory", "dynamic")),
                expected=int(raw.get("expected", 0)),
                predicted=int(raw.get("predicted", 0)),
                confidence=float(raw.get("confidence", 0.0)),
                difficulty=str(raw.get("difficulty", "medium")),
                source=str(raw.get("source", "dynamic")),
                tags=[str(t) for t in tags],
                run_seed=raw.get("run_seed"),
                input_preview=preview,
                timestamp=str(raw.get("timestamp", "")),
                error=raw.get("error"),
                record_hash=rec_hash,
            )
            records.append(record)
            category_counts[category] = category_counts.get(category, 0) + 1

    grouped: Dict[str, Dict[str, int]] = {}
    for rec in records:
        grouped.setdefault(rec.model, {})
        grouped[rec.model][rec.category] = grouped[rec.model].get(rec.category, 0) + 1

    return {
        "records": records,
        "total": len(records),
        "by_model_category": grouped,
        "source_files": per_model_files,
    }
