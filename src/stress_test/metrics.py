"""Metrics calculators for stress test evaluation."""
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MetricsResult:
    """Complete metrics from stress test."""
    accuracy: float
    fp_rate: float
    fn_rate: float
    precision: float
    recall: float
    f1_score: float
    
    # Calibration
    calibration_error: float  # Expected Calibration Error
    
    # Counts
    total_samples: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Thresholds
    accuracy_passed: bool = True
    fp_passed: bool = True
    fn_passed: bool = True
    calibration_passed: bool = True
    
    # Per-category breakdown
    category_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def all_passed(self) -> bool:
        return self.accuracy_passed and self.fp_passed and self.fn_passed and self.calibration_passed
    
    def get_failures(self) -> List[str]:
        failures = []
        if not self.accuracy_passed:
            failures.append('accuracy')
        if not self.fp_passed:
            failures.append('fp_rate')
        if not self.fn_passed:
            failures.append('fn_rate')
        if not self.calibration_passed:
            failures.append('calibration')
        return failures
    
    def get_warnings(self, warn_threshold: float = 0.8) -> List[str]:
        """Get metrics approaching threshold."""
        warnings = []
        # These would need threshold values passed in
        return warnings


class MetricsCalculator:
    """Calculate accuracy, FP, FN, and calibration metrics."""
    
    def __init__(self,
                 accuracy_threshold: float = 0.99,
                 fp_threshold: float = 0.03,
                 fn_threshold: float = 0.02,
                 calibration_threshold: float = 0.1):
        self.accuracy_threshold = accuracy_threshold
        self.fp_threshold = fp_threshold
        self.fn_threshold = fn_threshold
        self.calibration_threshold = calibration_threshold
    
    def calculate(self,
                  y_true: List[int],
                  y_pred: List[int],
                  y_prob: List[float] = None,
                  categories: List[str] = None) -> MetricsResult:
        """Calculate all metrics."""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic counts
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        
        total = len(y_true)
        total_pos = int(np.sum(y_true == 1))
        total_neg = int(np.sum(y_true == 0))
        
        # Metrics
        accuracy = (tp + tn) / total if total > 0 else 0
        fp_rate = fp / total_neg if total_neg > 0 else 0
        fn_rate = fn / total_pos if total_pos > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calibration (ECE)
        calibration_error = 0.0
        if y_prob is not None:
            calibration_error = self._expected_calibration_error(y_true, y_prob)
        
        # Per-category metrics
        category_metrics = {}
        if categories is not None:
            category_metrics = self._calculate_per_category(y_true, y_pred, categories)
        
        return MetricsResult(
            accuracy=accuracy,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
            precision=precision,
            recall=recall,
            f1_score=f1,
            calibration_error=calibration_error,
            total_samples=total,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy_passed=accuracy >= self.accuracy_threshold,
            fp_passed=fp_rate <= self.fp_threshold,
            fn_passed=fn_rate <= self.fn_threshold,
            calibration_passed=calibration_error <= self.calibration_threshold,
            category_metrics=category_metrics,
        )
    
    def _expected_calibration_error(self, y_true: np.ndarray, y_prob: List[float], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error."""
        y_prob = np.array(y_prob)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence = np.mean(y_prob[in_bin])
                avg_accuracy = np.mean(y_true[in_bin])
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return float(ece)
    
    def _calculate_per_category(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 categories: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per category."""
        unique_categories = list(set(categories))
        results = {}
        
        for cat in unique_categories:
            mask = np.array([c == cat for c in categories])
            if not np.any(mask):
                continue
            
            cat_true = y_true[mask]
            cat_pred = y_pred[mask]
            
            tp = int(np.sum((cat_true == 1) & (cat_pred == 1)))
            tn = int(np.sum((cat_true == 0) & (cat_pred == 0)))
            fp = int(np.sum((cat_true == 0) & (cat_pred == 1)))
            fn = int(np.sum((cat_true == 1) & (cat_pred == 0)))
            
            total = len(cat_true)
            total_neg = int(np.sum(cat_true == 0))
            total_pos = int(np.sum(cat_true == 1))
            
            results[cat] = {
                'accuracy': (tp + tn) / total if total > 0 else 0,
                'fp_rate': fp / total_neg if total_neg > 0 else 0,
                'fn_rate': fn / total_pos if total_pos > 0 else 0,
                'count': total,
            }
        
        return results
    
    def format_metrics(self, metrics: MetricsResult) -> str:
        """Format metrics as human-readable string."""
        lines = []
        
        # Main metrics table
        lines.append("┌────────────────┬─────────┬────────┬────────┐")
        lines.append("│ Metric         │ Value   │ Target │ Status │")
        lines.append("├────────────────┼─────────┼────────┼────────┤")
        
        acc_status = "✅" if metrics.accuracy_passed else "❌"
        lines.append(f"│ Accuracy       │ {metrics.accuracy*100:5.2f}%  │ >99%   │   {acc_status}   │")
        
        fp_status = "✅" if metrics.fp_passed else "❌"
        lines.append(f"│ FP Rate        │ {metrics.fp_rate*100:5.2f}%  │ <3%    │   {fp_status}   │")
        
        fn_status = "✅" if metrics.fn_passed else "❌"
        lines.append(f"│ FN Rate        │ {metrics.fn_rate*100:5.2f}%  │ <2%    │   {fn_status}   │")
        
        cal_status = "✅" if metrics.calibration_passed else "❌"
        lines.append(f"│ Calibration    │ {metrics.calibration_error:5.3f}   │ <0.1   │   {cal_status}   │")
        
        lines.append("└────────────────┴─────────┴────────┴────────┘")
        
        # Additional metrics
        lines.append(f"\nPrecision: {metrics.precision*100:.2f}%")
        lines.append(f"Recall: {metrics.recall*100:.2f}%")
        lines.append(f"F1 Score: {metrics.f1_score*100:.2f}%")
        lines.append(f"Total Samples: {metrics.total_samples:,}")
        
        # Confusion matrix
        lines.append(f"\nConfusion Matrix:")
        lines.append(f"  TP: {metrics.true_positives:,}  FP: {metrics.false_positives:,}")
        lines.append(f"  FN: {metrics.false_negatives:,}  TN: {metrics.true_negatives:,}")
        
        # Per-category breakdown
        if metrics.category_metrics:
            lines.append(f"\nPer-Category Breakdown:")
            for cat, cat_metrics in sorted(metrics.category_metrics.items()):
                fp_warn = "⚠️" if cat_metrics['fp_rate'] > 0.025 else ""
                lines.append(f"  {cat}: {cat_metrics['accuracy']*100:.1f}% acc, "
                           f"{cat_metrics['fp_rate']*100:.1f}% FP {fp_warn} "
                           f"(n={cat_metrics['count']})")
        
        return "\n".join(lines)


def calculate_metrics(y_true: List[int], 
                      y_pred: List[int],
                      y_prob: List[float] = None,
                      categories: List[str] = None) -> MetricsResult:
    """Convenience function to calculate metrics."""
    calc = MetricsCalculator()
    return calc.calculate(y_true, y_pred, y_prob, categories)


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 1000
    
    y_true = np.random.randint(0, 2, n).tolist()
    y_prob = [0.9 if y == 1 else 0.1 for y in y_true]
    # Add some noise
    y_prob = [min(1, max(0, p + np.random.normal(0, 0.15))) for p in y_prob]
    y_pred = [1 if p > 0.5 else 0 for p in y_prob]
    categories = np.random.choice(['email', 'code', 'text', 'url'], n).tolist()
    
    calc = MetricsCalculator()
    metrics = calc.calculate(y_true, y_pred, y_prob, categories)
    
    print(calc.format_metrics(metrics))
    print(f"\nAll passed: {metrics.all_passed()}")
    print(f"Failures: {metrics.get_failures()}")
