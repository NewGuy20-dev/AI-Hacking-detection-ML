"""Dataset packaging utilities for feedback-loop outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def write_model_dataset(output_dir: str | Path, model: str, samples: List[Dict]) -> Path:
    """Write model dataset JSONL and return output path."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"hard_examples_{model}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "id": sample.get("id"),
                "model": model,
                "text": sample.get("text", ""),
                "label": int(sample.get("label", 0)),
                "category": sample.get("category", "unknown"),
                "subcategory": sample.get("subcategory", "dynamic"),
                "difficulty": sample.get("difficulty", "medium"),
                "tags": sample.get("tags", []),
                "origin": sample.get("origin", "failure_loop"),
                "source_failure_id": sample.get("source_failure_id"),
                "run_seed": sample.get("run_seed"),
                "generator_seed": sample.get("generator_seed"),
                "created_at": sample.get("created_at"),
            }
            handle.write(json.dumps(payload) + "\n")
    return path


def write_manifest(output_dir: str | Path, manifest: Dict) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "candidate_dataset_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def write_summary(output_dir: str | Path, summary: Dict, filename: str = "loop_summary.json") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
