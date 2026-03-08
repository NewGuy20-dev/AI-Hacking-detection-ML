"""JSON logger for per-scenario logging in V1.4 stress test suite."""
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict

import numpy as np

from .scenarios import ScenarioResult


class JSONLogger:
    """Per-scenario JSONL logger with real-time category stats."""
    
    def __init__(self, output_dir: Path, model_name: str, run_date: str, run_seed: int = None):
        # Create date-based subfolder
        date_folder = output_dir / run_date
        self.output_path = date_folder / f"{model_name}_{run_date}.jsonl"
        self.failure_path = date_folder / f"{model_name}_{run_date}_failures.jsonl"
        self.model_name = model_name
        self.run_seed = run_seed
        self.file = None
        self.failure_file = None
        self.stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
        self.difficulty_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
        self.total_logged = 0
        self.include_input_summary = os.getenv("STRESS_LOG_INPUT_SUMMARY", "1") != "0"
        
    def __enter__(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.output_path, 'w', encoding='utf-8')
        self.failure_file = open(self.failure_path, 'w', encoding='utf-8')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        if self.failure_file:
            self.failure_file.close()
        
    def log(self, result: ScenarioResult):
        """Log a single scenario result."""
        record = {
            'scenario_id': result.scenario.id,
            'model': result.scenario.model,
            'category': result.scenario.category,
            'subcategory': result.scenario.subcategory,
            'input_preview': self._preview_input(result.scenario.input_data),
            'expected': result.scenario.expected_label,
            'predicted': result.prediction,
            'confidence': round(result.confidence, 4),
            'passed': result.passed,
            'latency_ms': round(result.latency_ms, 2),
            'difficulty': result.scenario.difficulty,
            'source': result.scenario.source,
            'tags': result.scenario.tags or [result.scenario.category],
            'run_seed': self.run_seed,
            'timestamp': result.timestamp,
            'error': result.error
        }
        metadata = result.metadata or {}
        if "threshold_used" in metadata:
            record["threshold_used"] = metadata["threshold_used"]
        if "prefiltered" in metadata:
            record["prefiltered"] = bool(metadata["prefiltered"])
        if "calibrated" in metadata:
            record["calibrated"] = bool(metadata["calibrated"])
        if metadata.get("model_artifact") is not None:
            record["model_artifact"] = metadata["model_artifact"]
        if metadata.get("raw_probability") is not None:
            record["raw_probability"] = round(float(metadata["raw_probability"]), 6)
        if metadata.get("calibrated_probability") is not None:
            record["calibrated_probability"] = round(float(metadata["calibrated_probability"]), 6)
        if self.include_input_summary:
            summary = self._summarize_input(result.scenario.input_data)
            if summary is not None:
                record['input_summary'] = summary

        serialized = json.dumps(record, default=self._json_default)
        self.file.write(serialized + '\n')
        self.file.flush()
        if not result.passed and self.failure_file:
            self.failure_file.write(serialized + '\n')
            self.failure_file.flush()
        
        # Update category stats
        cat = result.scenario.category
        self.stats[cat]['total'] += 1
        self.stats[cat]['passed' if result.passed else 'failed'] += 1
        
        # Update difficulty stats
        diff = result.scenario.difficulty
        self.difficulty_stats[diff]['total'] += 1
        self.difficulty_stats[diff]['passed' if result.passed else 'failed'] += 1
        
        self.total_logged += 1
        
    def _preview_input(self, input_data) -> str:
        """Create preview of input data."""
        if isinstance(input_data, str):
            return input_data[:100]
        elif isinstance(input_data, (list, tuple)):
            return f"[{len(input_data)} features]"
        elif hasattr(input_data, 'shape'):  # numpy array
            return f"[array shape: {input_data.shape}]"
        else:
            return f"[{type(input_data).__name__}]"

    def _summarize_input(self, input_data):
        """Optional compact numeric summary for debugging/profiling."""
        arr = None
        if isinstance(input_data, np.ndarray):
            arr = input_data.astype(np.float32, copy=False)
        elif isinstance(input_data, (list, tuple)):
            try:
                arr = np.asarray(input_data, dtype=np.float32)
            except (TypeError, ValueError):
                return None
        if arr is None or arr.size == 0:
            return None
        flat = arr.reshape(-1)
        return {
            "shape": list(arr.shape),
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "std": float(flat.std()),
            "p95": float(np.percentile(flat, 95)),
        }

    @staticmethod
    def _json_default(value):
        """Serialize numpy/path-like values to JSON-compatible Python types."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
    
    def get_category_accuracy(self) -> Dict[str, float]:
        """Returns accuracy per category for adaptive scheduling."""
        return {
            cat: s['passed'] / s['total'] 
            for cat, s in self.stats.items() 
            if s['total'] > 0
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        total = sum(s['total'] for s in self.stats.values())
        passed = sum(s['passed'] for s in self.stats.values())
        
        # Calculate per-difficulty accuracy
        accuracy_by_difficulty = {}
        for diff, stats in self.difficulty_stats.items():
            if stats['total'] > 0:
                accuracy_by_difficulty[diff] = stats['passed'] / stats['total']
        
        return {
            'model': self.model_name,
            'total_scenarios': total,
            'passed': passed,
            'failed': total - passed,
            'accuracy': passed / total if total > 0 else 0,
            'categories': dict(self.stats),
            'accuracy_by_difficulty': accuracy_by_difficulty,
            'difficulty_breakdown': dict(self.difficulty_stats)
        }
