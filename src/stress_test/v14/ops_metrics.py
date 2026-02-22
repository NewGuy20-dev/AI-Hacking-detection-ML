"""Streaming ops metrics for v1.4 stress tests."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import math
import numpy as np


@dataclass
class OpsMetricsState:
    """Accumulates confusion, calibration, and latency statistics."""
    # confusion counts
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # latency samples (milliseconds)
    latency: List[float] = field(default_factory=list)

    # per-category confusion
    per_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # per-difficulty confusion
    per_difficulty: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # calibration bins (ECE)
    bins: int = 10
    bin_totals: Optional[np.ndarray] = None
    bin_conf_sum: Optional[np.ndarray] = None
    bin_pos_sum: Optional[np.ndarray] = None

    def __post_init__(self):
        self.bin_totals = np.zeros(self.bins, dtype=float)
        self.bin_conf_sum = np.zeros(self.bins, dtype=float)
        self.bin_pos_sum = np.zeros(self.bins, dtype=float)

    def _accum_cat(self, table: Dict[str, Dict[str, int]], key: str, pred: int, truth: int):
        if key not in table:
            table[key] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        if truth == 1 and pred == 1:
            table[key]['tp'] += 1
        elif truth == 0 and pred == 0:
            table[key]['tn'] += 1
        elif truth == 0 and pred == 1:
            table[key]['fp'] += 1
        elif truth == 1 and pred == 0:
            table[key]['fn'] += 1

    def update(self, expected: int, predicted: int, confidence: float, latency_ms: float,
               category: Optional[str] = None, difficulty: Optional[str] = None):
        if expected == 1 and predicted == 1:
            self.tp += 1
        elif expected == 0 and predicted == 0:
            self.tn += 1
        elif expected == 0 and predicted == 1:
            self.fp += 1
        elif expected == 1 and predicted == 0:
            self.fn += 1

        if category:
            self._accum_cat(self.per_category, category, predicted, expected)
        if difficulty:
            self._accum_cat(self.per_difficulty, difficulty, predicted, expected)

        if confidence is not None:
            conf = max(0.0, min(1.0, float(confidence)))
            bin_idx = min(self.bins - 1, int(conf * self.bins))
            self.bin_totals[bin_idx] += 1
            self.bin_conf_sum[bin_idx] += conf
            self.bin_pos_sum[bin_idx] += (1 if expected == 1 else 0)

        if latency_ms is not None:
            self.latency.append(float(latency_ms))

    def _ece(self) -> float:
        totals = self.bin_totals
        if totals.sum() == 0:
            return 0.0
        conf_avg = np.divide(self.bin_conf_sum, totals, out=np.zeros_like(totals), where=totals>0)
        acc_avg = np.divide(self.bin_pos_sum, totals, out=np.zeros_like(totals), where=totals>0)
        gaps = np.abs(conf_avg - acc_avg)
        weights = totals / totals.sum()
        return float(np.sum(gaps * weights))

    def _latency_stats(self):
        if not self.latency:
            return {
                'p50_ms': 0.0,
                'p95_ms': 0.0,
                'p99_ms': 0.0,
                'mean_ms': 0.0,
                'throughput_sps': 0.0,
            }
        arr = np.array(self.latency, dtype=float)
        return {
            'p50_ms': float(np.percentile(arr, 50)),
            'p95_ms': float(np.percentile(arr, 95)),
            'p99_ms': float(np.percentile(arr, 99)),
            'mean_ms': float(np.mean(arr)),
            'throughput_sps': float(len(arr) / (np.sum(arr) / 1000.0)) if arr.sum() > 0 else 0.0,
        }

    def summary(self) -> Dict:
        total = self.tp + self.tn + self.fp + self.fn
        pos = self.tp + self.fn
        neg = self.tn + self.fp
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / pos if pos else 0.0
        fpr = self.fp / neg if neg else 0.0
        fnr = self.fn / pos if pos else 0.0
        accuracy = (self.tp + self.tn) / total if total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        lat = self._latency_stats()
        return {
            'counts': {'tp': self.tp, 'tn': self.tn, 'fp': self.fp, 'fn': self.fn, 'total': total},
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fp_rate': fpr,
                'fn_rate': fnr,
                'ece': self._ece(),
            },
            'latency': lat,
            'per_category': self.per_category,
            'per_difficulty': self.per_difficulty,
        }
