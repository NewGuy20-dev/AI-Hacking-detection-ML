"""Metrics tracker for accuracy, recall, precision, and FP rate computation."""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    fp_rate: float
    fn_rate: float
    tp: int
    tn: int
    fp: int
    fn: int
    total: int
    threshold: float = 0.5
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class MetricsTracker:
    """Track and compute classification metrics."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("evaluation/metrics_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: List[MetricsResult] = []
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None, 
                threshold: float = 0.5) -> MetricsResult:
        """Compute all metrics from predictions."""
        y_true = np.asarray(y_true)
        
        # If probabilities provided, apply threshold
        if y_prob is not None:
            y_pred = (np.asarray(y_prob) >= threshold).astype(int)
        else:
            y_pred = np.asarray(y_pred)
        
        # Confusion matrix components
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        total = len(y_true)
        
        # Metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return MetricsResult(
            accuracy=round(accuracy, 6),
            precision=round(precision, 6),
            recall=round(recall, 6),
            f1_score=round(f1, 6),
            fp_rate=round(fp_rate, 6),
            fn_rate=round(fn_rate, 6),
            tp=tp, tn=tn, fp=fp, fn=fn,
            total=total,
            threshold=threshold
        )
    
    @staticmethod
    def compute_at_thresholds(y_true: np.ndarray, y_prob: np.ndarray, 
                               thresholds: List[float] = None) -> Dict[float, MetricsResult]:
        """Compute metrics at multiple thresholds."""
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        results = {}
        for t in thresholds:
            results[t] = MetricsTracker.compute(y_true, None, y_prob, threshold=t)
        return results
    
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                                target_recall: float = 0.98,
                                max_fp_rate: float = 0.03) -> tuple:
        """Find threshold that achieves target recall with lowest FP rate."""
        best_threshold = 0.5
        best_fp_rate = 1.0
        
        for t in np.arange(0.05, 0.95, 0.01):
            metrics = MetricsTracker.compute(y_true, None, y_prob, threshold=t)
            if metrics.recall >= target_recall and metrics.fp_rate <= max_fp_rate:
                if metrics.fp_rate < best_fp_rate:
                    best_fp_rate = metrics.fp_rate
                    best_threshold = t
        
        return best_threshold, MetricsTracker.compute(y_true, None, y_prob, threshold=best_threshold)
    
    def log(self, metrics: MetricsResult, model_name: str = "default"):
        """Log metrics to file."""
        self.history.append(metrics)
        
        log_file = self.log_dir / f"{model_name}_metrics.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(metrics)) + "\n")
    
    def get_summary(self) -> Dict:
        """Get summary of tracked metrics."""
        if not self.history:
            return {}
        
        return {
            "count": len(self.history),
            "avg_accuracy": np.mean([m.accuracy for m in self.history]),
            "avg_recall": np.mean([m.recall for m in self.history]),
            "avg_fp_rate": np.mean([m.fp_rate for m in self.history]),
            "best_accuracy": max(m.accuracy for m in self.history),
            "best_recall": max(m.recall for m in self.history),
            "lowest_fp_rate": min(m.fp_rate for m in self.history),
        }
    
    @staticmethod
    def format_report(metrics: MetricsResult, model_name: str = "") -> str:
        """Format metrics as readable report."""
        header = f"=== {model_name} Metrics ===" if model_name else "=== Metrics ==="
        return f"""
{header}
Accuracy:  {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)
Precision: {metrics.precision:.4f}
Recall:    {metrics.recall:.4f} ({metrics.recall*100:.2f}%)
F1 Score:  {metrics.f1_score:.4f}
FP Rate:   {metrics.fp_rate:.4f} ({metrics.fp_rate*100:.2f}%)
FN Rate:   {metrics.fn_rate:.4f}
Threshold: {metrics.threshold}

Confusion Matrix:
  TP: {metrics.tp:,}  FN: {metrics.fn:,}
  FP: {metrics.fp:,}  TN: {metrics.tn:,}
  Total: {metrics.total:,}
"""


def check_targets(metrics: MetricsResult, 
                  target_accuracy: float = 0.999,
                  target_recall: float = 0.98,
                  max_fp_rate: float = 0.03) -> Dict[str, bool]:
    """Check if metrics meet target thresholds."""
    return {
        "accuracy_met": metrics.accuracy >= target_accuracy,
        "recall_met": metrics.recall >= target_recall,
        "fp_rate_met": metrics.fp_rate <= max_fp_rate,
        "all_met": (metrics.accuracy >= target_accuracy and 
                   metrics.recall >= target_recall and 
                   metrics.fp_rate <= max_fp_rate)
    }
