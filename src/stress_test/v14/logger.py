"""JSON logger for per-scenario logging in V1.4 stress test suite."""
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict
from .scenarios import ScenarioResult


class JSONLogger:
    """Per-scenario JSONL logger with real-time category stats."""
    
    def __init__(self, output_dir: Path, model_name: str, run_date: str, run_seed: int = None):
        self.output_path = output_dir / f"{model_name}_{run_date}.jsonl"
        self.failure_path = output_dir / f"{model_name}_{run_date}_failures.jsonl"
        self.model_name = model_name
        self.run_seed = run_seed
        self.file = None
        self.failure_file = None
        self.stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
        self.difficulty_stats = defaultdict(lambda: {'total': 0, 'passed': 0, 'failed': 0})
        self.total_logged = 0
        
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
        
        self.file.write(json.dumps(record) + '\n')
        self.file.flush()
        if not result.passed and self.failure_file:
            self.failure_file.write(json.dumps(record) + '\n')
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
