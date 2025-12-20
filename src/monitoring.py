"""Phase D3: Monitoring and logging module."""
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import deque
from threading import Lock
import numpy as np


class PredictionMonitor:
    """Monitor prediction latency, throughput, and distribution drift."""
    
    def __init__(self, log_dir='logs', window_size=1000):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self._lock = Lock()
        
        # Setup logging
        self.logger = logging.getLogger('prediction_monitor')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_dir / 'predictions.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.logger.addHandler(handler)
        
        # Baseline stats for drift detection
        self.baseline_mean = 0.5
        self.baseline_std = 0.2
    
    def record(self, prediction, latency_ms, metadata=None):
        """Record a prediction for monitoring."""
        with self._lock:
            self.latencies.append(latency_ms)
            self.predictions.append(prediction)
            self.timestamps.append(time.time())
        
        # Log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': float(prediction),
            'latency_ms': latency_ms,
            'metadata': metadata or {}
        }
        self.logger.info(json.dumps(log_entry))
    
    def get_stats(self):
        """Get current monitoring statistics."""
        with self._lock:
            if not self.latencies:
                return {'status': 'no_data'}
            
            latencies = np.array(self.latencies)
            predictions = np.array(self.predictions)
            
            return {
                'latency': {
                    'mean_ms': float(np.mean(latencies)),
                    'p50_ms': float(np.percentile(latencies, 50)),
                    'p95_ms': float(np.percentile(latencies, 95)),
                    'p99_ms': float(np.percentile(latencies, 99)),
                },
                'predictions': {
                    'mean': float(np.mean(predictions)),
                    'std': float(np.std(predictions)),
                    'positive_rate': float((predictions > 0.5).mean()),
                },
                'throughput': {
                    'samples': len(self.latencies),
                    'window_seconds': self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0,
                },
                'drift_detected': self._check_drift(predictions)
            }
    
    def _check_drift(self, predictions):
        """Simple drift detection based on mean shift."""
        if len(predictions) < 100:
            return False
        current_mean = np.mean(predictions)
        return abs(current_mean - self.baseline_mean) > 2 * self.baseline_std
    
    def set_baseline(self, mean, std):
        """Set baseline statistics for drift detection."""
        self.baseline_mean = mean
        self.baseline_std = std


class AlertLogger:
    """Log security alerts with severity levels."""
    
    SEVERITY_LEVELS = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
    
    def __init__(self, log_dir='alerts'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('alert_logger')
        self.logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_dir / f'alerts_{datetime.now():%Y%m%d}.json')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
    
    def log_alert(self, severity, attack_type, confidence, source=None, details=None):
        """Log a security alert."""
        alert = {
            'id': f"ALERT-{int(time.time()*1000)}",
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'attack_type': attack_type,
            'confidence': confidence,
            'source': source,
            'details': details or {}
        }
        self.logger.info(json.dumps(alert))
        
        if self.SEVERITY_LEVELS.get(severity, 0) >= 3:
            print(f"⚠️  {severity} ALERT: {attack_type} (confidence: {confidence:.1%})")
        
        return alert['id']


class InferenceLogger:
    """Structured logging for inference requests."""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.monitor = PredictionMonitor(log_dir)
        self.alert_logger = AlertLogger()
    
    def log_inference(self, input_type, input_data, result, latency_ms):
        """Log an inference request with timing."""
        confidence = result.get('confidence', [0.5])
        confidence = confidence[0] if hasattr(confidence, '__len__') else confidence
        
        self.monitor.record(confidence, latency_ms, {'type': input_type})
        
        # Auto-alert on high confidence attacks
        if confidence > 0.8:
            self.alert_logger.log_alert(
                severity='HIGH' if confidence > 0.9 else 'MEDIUM',
                attack_type=input_type,
                confidence=confidence,
                details={'input_preview': str(input_data)[:100]}
            )
        
        return self.monitor.get_stats()


# Global instances
_monitor = None
_inference_logger = None


def get_monitor():
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor


def get_inference_logger():
    global _inference_logger
    if _inference_logger is None:
        _inference_logger = InferenceLogger()
    return _inference_logger
