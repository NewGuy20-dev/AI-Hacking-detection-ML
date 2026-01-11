"""Confidence calibration for accurate probability estimates."""
import numpy as np
from typing import Optional, Tuple
from pathlib import Path
import json


class ConfidenceCalibrator:
    """Calibrate model confidence scores using Platt scaling or isotonic regression."""
    
    def __init__(self, method: str = "platt"):
        """Initialize calibrator.
        
        Args:
            method: 'platt' for Platt scaling, 'isotonic' for isotonic regression
        """
        self.method = method
        self.params = {}
        self.fitted = False
    
    def fit(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit calibration parameters."""
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob).clip(1e-7, 1 - 1e-7)
        
        if self.method == "platt":
            self._fit_platt(y_true, y_prob)
        elif self.method == "isotonic":
            self._fit_isotonic(y_true, y_prob)
        
        self.fitted = True
    
    def _fit_platt(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit Platt scaling (logistic regression on log-odds)."""
        # Convert to log-odds
        log_odds = np.log(y_prob / (1 - y_prob))
        
        # Simple gradient descent for A, B in sigmoid(A * log_odds + B)
        A, B = 1.0, 0.0
        lr = 0.01
        
        for _ in range(1000):
            z = A * log_odds + B
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            
            # Gradient
            error = p - y_true
            grad_A = np.mean(error * log_odds)
            grad_B = np.mean(error)
            
            A -= lr * grad_A
            B -= lr * grad_B
        
        self.params = {"A": float(A), "B": float(B)}
    
    def _fit_isotonic(self, y_true: np.ndarray, y_prob: np.ndarray):
        """Fit isotonic regression (piecewise constant monotonic)."""
        # Sort by probability
        order = np.argsort(y_prob)
        y_prob_sorted = y_prob[order]
        y_true_sorted = y_true[order]
        
        # Pool Adjacent Violators Algorithm (simplified)
        n = len(y_prob)
        bins = 20
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_probs = []
        
        for i in range(bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_probs.append((bin_edges[i], bin_edges[i + 1], y_true[mask].mean()))
            else:
                bin_probs.append((bin_edges[i], bin_edges[i + 1], bin_edges[i]))
        
        self.params = {"bins": bin_probs}
    
    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to probability scores."""
        if not self.fitted:
            return y_prob
        
        y_prob = np.asarray(y_prob).clip(1e-7, 1 - 1e-7)
        
        if self.method == "platt":
            log_odds = np.log(y_prob / (1 - y_prob))
            z = self.params["A"] * log_odds + self.params["B"]
            return 1 / (1 + np.exp(-z))
        
        elif self.method == "isotonic":
            result = np.zeros_like(y_prob)
            for i, p in enumerate(y_prob):
                for low, high, cal_p in self.params["bins"]:
                    if low <= p < high:
                        result[i] = cal_p
                        break
                else:
                    result[i] = p
            return result
        
        return y_prob
    
    def save(self, path: Path):
        """Save calibration parameters."""
        with open(path, "w") as f:
            json.dump({"method": self.method, "params": self.params}, f, indent=2)
    
    def load(self, path: Path):
        """Load calibration parameters."""
        with open(path, "r") as f:
            data = json.load(f)
        self.method = data["method"]
        self.params = data["params"]
        self.fitted = True


class EnsembleCalibrator:
    """Calibrate ensemble of models."""
    
    def __init__(self):
        self.calibrators = {}
    
    def fit_model(self, model_name: str, y_true: np.ndarray, y_prob: np.ndarray, method: str = "platt"):
        """Fit calibrator for a specific model."""
        cal = ConfidenceCalibrator(method=method)
        cal.fit(y_true, y_prob)
        self.calibrators[model_name] = cal
    
    def calibrate_model(self, model_name: str, y_prob: np.ndarray) -> np.ndarray:
        """Calibrate predictions for a specific model."""
        if model_name in self.calibrators:
            return self.calibrators[model_name].calibrate(y_prob)
        return y_prob
    
    def calibrate_ensemble(self, predictions: dict) -> dict:
        """Calibrate all model predictions in ensemble."""
        return {name: self.calibrate_model(name, prob) for name, prob in predictions.items()}
    
    def save(self, dir_path: Path):
        """Save all calibrators."""
        dir_path.mkdir(parents=True, exist_ok=True)
        for name, cal in self.calibrators.items():
            cal.save(dir_path / f"{name}_calibration.json")
    
    def load(self, dir_path: Path):
        """Load all calibrators."""
        for f in dir_path.glob("*_calibration.json"):
            name = f.stem.replace("_calibration", "")
            cal = ConfidenceCalibrator()
            cal.load(f)
            self.calibrators[name] = cal


def compute_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    
    return ece / len(y_true)
