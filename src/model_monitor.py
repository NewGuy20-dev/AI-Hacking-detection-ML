"""Runtime monitoring and drift detection for ML models."""
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class ModelMonitor:
    """Collects runtime metrics for model predictions."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._start_time: Optional[float] = None
    
    def log_prediction(self, model_name: str, latency: float, confidence: float):
        """Log a single prediction's metrics."""
        self._metrics[f"{model_name}_latency"].append(latency)
        self._metrics[f"{model_name}_confidence"].append(confidence)
        # Trim if exceeds max
        for key in [f"{model_name}_latency", f"{model_name}_confidence"]:
            if len(self._metrics[key]) > self.max_samples:
                self._metrics[key] = self._metrics[key][-self.max_samples:]
    
    @contextmanager
    def track(self, model_name: str):
        """Context manager to track prediction latency."""
        start = time.perf_counter()
        result = {'confidence': None}
        yield result
        latency = time.perf_counter() - start
        conf = result.get('confidence', 0.5)
        if isinstance(conf, (list, np.ndarray)):
            conf = float(np.mean(conf))
        self.log_prediction(model_name, latency, conf)
    
    def get_stats(self, model_name: str = None) -> Dict:
        """Get statistics for a model or all models."""
        if model_name:
            return self._compute_stats(model_name)
        return {name.rsplit('_', 1)[0]: self._compute_stats(name.rsplit('_', 1)[0]) 
                for name in self._metrics if name.endswith('_latency')}
    
    def _compute_stats(self, model_name: str) -> Dict:
        latencies = self._metrics.get(f"{model_name}_latency", [])
        confidences = self._metrics.get(f"{model_name}_confidence", [])
        if not latencies:
            return {}
        return {
            'latency_mean': float(np.mean(latencies)),
            'latency_p95': float(np.percentile(latencies, 95)) if len(latencies) >= 20 else None,
            'confidence_mean': float(np.mean(confidences)) if confidences else None,
            'count': len(latencies)
        }
    
    def reset(self, model_name: str = None):
        """Reset metrics for a model or all models."""
        if model_name:
            self._metrics.pop(f"{model_name}_latency", None)
            self._metrics.pop(f"{model_name}_confidence", None)
        else:
            self._metrics.clear()
    
    def export(self, filepath: str):
        """Export metrics to JSON file."""
        Path(filepath).write_text(json.dumps(self.get_stats(), indent=2))


class DriftDetector:
    """Detects data drift using statistical tests."""
    
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        self.reference = np.asarray(reference_data).flatten()
        self.threshold = threshold
        self._last_result = None
    
    def detect(self, new_data: np.ndarray) -> bool:
        """Detect drift using Kolmogorov-Smirnov test."""
        from scipy.stats import ks_2samp
        new_data = np.asarray(new_data).flatten()
        if len(new_data) < 10 or len(self.reference) < 10:
            return False
        statistic, p_value = ks_2samp(self.reference, new_data)
        self._last_result = {'statistic': statistic, 'p_value': p_value, 
                            'is_drifted': p_value < self.threshold}
        return p_value < self.threshold
    
    def get_report(self) -> Optional[Dict]:
        """Get the last drift detection report."""
        return self._last_result
    
    def update_reference(self, new_reference: np.ndarray):
        """Update the reference distribution."""
        self.reference = np.asarray(new_reference).flatten()
        self._last_result = None
