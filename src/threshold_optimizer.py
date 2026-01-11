"""Threshold optimizer for finding optimal classification thresholds."""
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from metrics_tracker import MetricsTracker, MetricsResult


@dataclass
class ThresholdResult:
    """Result of threshold optimization."""
    threshold: float
    accuracy: float
    recall: float
    precision: float
    fp_rate: float
    f1_score: float


class ThresholdOptimizer:
    """Optimize classification thresholds for target metrics."""
    
    def __init__(self, target_recall: float = 0.98, max_fp_rate: float = 0.03):
        self.target_recall = target_recall
        self.max_fp_rate = max_fp_rate
    
    def grid_search(self, y_true: np.ndarray, y_prob: np.ndarray,
                    thresholds: Optional[List[float]] = None) -> List[ThresholdResult]:
        """Perform grid search over thresholds."""
        if thresholds is None:
            thresholds = np.arange(0.05, 0.95, 0.01).tolist()
        
        results = []
        for t in thresholds:
            metrics = MetricsTracker.compute(y_true, None, y_prob, threshold=t)
            results.append(ThresholdResult(
                threshold=round(t, 3),
                accuracy=metrics.accuracy,
                recall=metrics.recall,
                precision=metrics.precision,
                fp_rate=metrics.fp_rate,
                f1_score=metrics.f1_score
            ))
        return results
    
    def find_optimal(self, y_true: np.ndarray, y_prob: np.ndarray,
                     strategy: str = "recall_first") -> Tuple[float, ThresholdResult]:
        """Find optimal threshold based on strategy.
        
        Strategies:
        - recall_first: Maximize recall while keeping FP rate under limit
        - balanced: Maximize F1 score
        - fp_first: Minimize FP rate while keeping recall above target
        """
        results = self.grid_search(y_true, y_prob)
        
        if strategy == "recall_first":
            # Find lowest threshold that achieves target recall with acceptable FP
            valid = [r for r in results if r.recall >= self.target_recall and r.fp_rate <= self.max_fp_rate]
            if valid:
                best = min(valid, key=lambda x: x.fp_rate)
            else:
                # Fallback: best recall under FP limit
                valid = [r for r in results if r.fp_rate <= self.max_fp_rate]
                best = max(valid, key=lambda x: x.recall) if valid else results[len(results)//2]
        
        elif strategy == "balanced":
            best = max(results, key=lambda x: x.f1_score)
        
        elif strategy == "fp_first":
            valid = [r for r in results if r.recall >= self.target_recall]
            if valid:
                best = min(valid, key=lambda x: x.fp_rate)
            else:
                best = max(results, key=lambda x: x.recall)
        
        else:
            best = results[len(results)//2]
        
        return best.threshold, best
    
    def find_pareto_optimal(self, y_true: np.ndarray, y_prob: np.ndarray) -> List[ThresholdResult]:
        """Find Pareto-optimal thresholds (no threshold dominates another)."""
        results = self.grid_search(y_true, y_prob)
        pareto = []
        
        for r in results:
            dominated = False
            for other in results:
                # Check if other dominates r (better in all metrics)
                if (other.recall >= r.recall and other.fp_rate <= r.fp_rate and
                    (other.recall > r.recall or other.fp_rate < r.fp_rate)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)
        
        return sorted(pareto, key=lambda x: x.threshold)
    
    def optimize_per_attack_type(self, data: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Dict[str, float]:
        """Optimize thresholds per attack type."""
        thresholds = {}
        for attack_type, (y_true, y_prob) in data.items():
            thresh, _ = self.find_optimal(y_true, y_prob, strategy="recall_first")
            thresholds[attack_type] = thresh
        return thresholds


def save_optimal_thresholds(thresholds: Dict[str, float], output_path: Path):
    """Save optimal thresholds to config file."""
    config = {
        "thresholds": thresholds,
        "metadata": {
            "target_recall": 0.98,
            "max_fp_rate": 0.03,
            "strategy": "recall_first"
        }
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)


def load_optimal_thresholds(config_path: Path) -> Dict[str, float]:
    """Load optimal thresholds from config file."""
    if not config_path.exists():
        return {"default": 0.5}
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("thresholds", {"default": 0.5})
