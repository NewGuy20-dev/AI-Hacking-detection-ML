"""Replay loader for real-world evaluation sources."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import json
import csv


@dataclass
class ReplaySample:
    input_data: Any
    expected_label: int
    category: str
    source: str


def _load_jsonl(path: Path, text_field: str, label_field: str, label_map: Dict[str, int], max_samples: int) -> List[ReplaySample]:
    samples: List[ReplaySample] = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            try:
                rec = json.loads(line)
                txt = rec.get(text_field)
                lab_raw = rec.get(label_field)
                if txt is None or lab_raw is None:
                    continue
                lab = label_map.get(str(lab_raw), label_map.get(lab_raw, lab_raw))
                lab = int(lab)
                samples.append(ReplaySample(txt, lab, rec.get('category', 'realworld'), path.stem))
            except Exception:
                continue
    return samples


def _load_csv(path: Path, text_field: str, label_field: str, label_map: Dict[str, int], max_samples: int) -> List[ReplaySample]:
    samples: List[ReplaySample] = []
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(samples) >= max_samples:
                break
            txt = row.get(text_field)
            lab_raw = row.get(label_field)
            if txt is None or lab_raw is None:
                continue
            lab = label_map.get(str(lab_raw), label_map.get(lab_raw, lab_raw))
            lab = int(lab)
            samples.append(ReplaySample(txt, lab, row.get('category', 'realworld'), path.stem))
    return samples


def load_replay_source(entry: Dict[str, Any]) -> List[ReplaySample]:
    if not entry.get('enabled', True):
        return []
    path = Path(entry['path'])
    if not path.exists():
        raise FileNotFoundError(f"Replay source not found: {path}")
    fmt = entry.get('format', 'jsonl').lower()
    text_field = entry.get('text_field', 'text')
    label_field = entry.get('label_field', 'label')
    label_map = entry.get('label_map', {}) or {}
    max_samples = int(entry.get('max_samples', 10000))
    if fmt == 'jsonl':
        return _load_jsonl(path, text_field, label_field, label_map, max_samples)
    if fmt == 'csv':
        return _load_csv(path, text_field, label_field, label_map, max_samples)
    raise ValueError(f"Unsupported replay format: {fmt}")
