# Phase B: Production Features Implementation Plan

## Overview

This document details the implementation plan for production-ready features including Alert System, A/B Testing, Automated Retraining, Gradio Dashboard, and API Server.

**Total Estimated Work:**
- Files: ~25
- Lines of Code: ~1,400
- Implementation Time: 5-7 hours

---

## Codebase Analysis Summary

### Existing Infrastructure (to leverage)

| Component | Location | Status | Reuse Strategy |
|-----------|----------|--------|----------------|
| `AlertManager` | `src/alert_manager.py` | ✅ Working | Extend with channels |
| `BatchHybridPredictor` | `src/batch_predictor.py` | ✅ Working | Add alert integration |
| `ModelRegistry` | `src/model_registry.py` | ✅ Working | Use for A/B versioning |
| `ModelMonitor` | `src/model_monitor.py` | ✅ Working | Use for A/B metrics |
| `DriftDetector` | `src/model_monitor.py` | ✅ Working | Use in retraining trigger |
| `InputValidator` | `src/input_validator.py` | ✅ Working | Use in API/Dashboard |
| `PayloadExplainer` | `src/explainability_v2.py` | ✅ Working | Use in Dashboard |
| `HybridPredictor` | `src/hybrid_predictor.py` | ✅ Working | Use in API/Dashboard |

### Current Model Performance

| Model | File | Accuracy | Notes |
|-------|------|----------|-------|
| Payload CNN | `models/payload_cnn.pt` | 99.89% | 3 false positives on edge cases |
| URL CNN | `models/url_cnn.pt` | 97.47% | 100% on validation suite |
| TimeSeries LSTM | `models/timeseries_lstm.pt` | 75.38% | Needs improvement |
| Network RF | `models/network_intrusion_model.pkl` | ~95% | sklearn RandomForest |
| Fraud XGB | `models/fraud_detection_model.pkl` | ~94% | sklearn XGBoost |

### Validation Results (from `evaluation/validation_report.json`)
- **Total**: 92.9% (39/42 tests passed)
- **Payload**: 89.3% (25/28) - 3 false positives on benign edge cases
- **URL**: 100% (14/14)

---

## Table of Contents

1. [Feature 1: Alert System](#feature-1-alert-system)
2. [Feature 2: A/B Testing Framework](#feature-2-ab-testing-framework)
3. [Feature 3: Automated Retraining Trigger](#feature-3-automated-retraining-trigger)
4. [Feature 4: Gradio Dashboard](#feature-4-gradio-dashboard)
5. [Feature 5: API Server](#feature-5-api-server)
6. [Integration Map](#integration-map)
7. [Implementation Order](#implementation-order)
8. [File Summary](#file-summary)
9. [Requirements](#requirements)
10. [Testing Strategy](#testing-strategy)

---

## Feature 1: Alert System

### Purpose
Extend existing `AlertManager` with multi-channel notifications (webhook, email, console) and integrate with prediction pipeline.

### Existing Code to Extend

**Current `src/alert_manager.py`** provides:
- `Severity` enum (LOW, MEDIUM, HIGH, CRITICAL)
- `AlertManager.create_alert()` - creates structured alerts
- `AlertManager.export_json()` - exports to JSON file
- `AlertManager.export_syslog()` - formats for syslog

**Gap**: No external notification channels, no rate limiting, no integration with predictor.

### Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `src/alerts/__init__.py` | 10 | Package exports + AlertDispatcher |
| `src/alerts/dispatcher.py` | 100 | Main AlertDispatcher class |
| `src/alerts/channels/__init__.py` | 5 | Channel exports |
| `src/alerts/channels/base.py` | 30 | BaseChannel abstract class |
| `src/alerts/channels/webhook.py` | 55 | Slack/Discord/Teams webhook |
| `src/alerts/channels/email.py` | 60 | SMTP email notifications |
| `src/alerts/channels/console.py` | 25 | Console/log output |
| `config/alerts.yaml` | 40 | Configuration |
| `tests/test_alerts.py` | 80 | Unit tests |

### Class: AlertDispatcher

**Location:** `src/alerts/dispatcher.py`

**Integration with existing code:**
- Uses `Severity` enum from `src/alert_manager.py`
- Compatible with alert dict format from `AlertManager.create_alert()`

```python
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import yaml
import time

from src.alert_manager import Severity

class AlertDispatcher:
    """Routes alerts to configured channels based on severity."""
    
    def __init__(self, config_path: str = "config/alerts.yaml"):
        """Load configuration and initialize channels."""
        self.config = self._load_config(config_path)
        self.channels = {}
        self.routing = {}
        self._rate_limiter = defaultdict(list)
        self._alert_counter = 0
        self._init_channels()
    
    def _load_config(self, path: str) -> dict:
        """Load YAML config with env var substitution."""
    
    def _init_channels(self):
        """Initialize channels from config."""
    
    def send(self, alert: dict) -> dict:
        """
        Send alert to appropriate channels based on severity.
        
        Args:
            alert: Dict from AlertManager.create_alert() or custom:
                {
                    "severity": "HIGH",
                    "attack_type": "SQL_INJECTION",
                    "confidence": 0.95,
                    "source": {"payload": "...", "ip": "..."},
                    "timestamp": "2025-12-21T05:00:00Z"
                }
        
        Returns:
            {"success": True, "alert_id": "ALT-...", "channels_notified": ["webhook"]}
        """
    
    def send_from_prediction(self, prediction: dict, input_data: dict, 
                             threshold: float = 0.8) -> dict:
        """
        Create and send alert from BatchHybridPredictor result.
        
        Args:
            prediction: Result from predictor.predict_batch()
            input_data: Original input (payloads, urls, etc.)
            threshold: Confidence threshold for alerting
        """
    
    def add_channel(self, name: str, channel: 'BaseChannel'):
        """Register a new notification channel."""
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID like ALT-20251221-001."""
    
    def _check_rate_limit(self, alert_type: str) -> bool:
        """Check if rate limit exceeded for this alert type."""
    
    def _aggregate_similar(self, alert: dict) -> dict:
        """Aggregate similar alerts within time window."""
```

### Class: BaseChannel

**Location:** `src/alerts/channels/base.py`

```python
from abc import ABC, abstractmethod

class BaseChannel(ABC):
    """Abstract base class for notification channels."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.retry_attempts = 3
        self.timeout = 10
    
    @abstractmethod
    def send(self, alert: dict) -> bool:
        """Send alert through this channel. Returns success status."""
    
    @abstractmethod
    def format_message(self, alert: dict) -> str:
        """Format alert for this specific channel."""
    
    def is_enabled(self) -> bool:
        """Check if channel is enabled."""
```

### Class: WebhookChannel

**Location:** `src/alerts/channels/webhook.py`

```python
class WebhookChannel(BaseChannel):
    """Send alerts via webhook (Slack, Discord, Teams)."""
    
    def __init__(self, url: str, format: str = "slack"):
        """
        Args:
            url: Webhook URL
            format: "slack", "discord", or "teams"
        """
    
    def send(self, alert: dict) -> bool:
        """POST alert to webhook URL with retries."""
    
    def format_message(self, alert: dict) -> str:
        """Format as Slack/Discord/Teams message."""
    
    def _format_slack(self, alert: dict) -> dict:
        """Slack Block Kit format."""
    
    def _format_discord(self, alert: dict) -> dict:
        """Discord embed format."""
```

### Class: EmailChannel

**Location:** `src/alerts/channels/email.py`

```python
class EmailChannel(BaseChannel):
    """Send alerts via SMTP email."""
    
    def __init__(self, smtp_host: str, smtp_port: int, 
                 username: str, password: str, recipients: list):
        """Initialize SMTP connection settings."""
    
    def send(self, alert: dict) -> bool:
        """Send email with TLS."""
    
    def format_message(self, alert: dict) -> str:
        """Format as HTML email."""
```

### Class: ConsoleChannel

**Location:** `src/alerts/channels/console.py`

```python
class ConsoleChannel(BaseChannel):
    """Print alerts to console/log."""
    
    def send(self, alert: dict) -> bool:
        """Print formatted alert to stdout."""
    
    def format_message(self, alert: dict) -> str:
        """Format as colored console output."""
```

### Alert Structure

```python
{
    "id": "ALT-20251221-001",
    "timestamp": "2025-12-21T05:00:00Z",
    "severity": "HIGH",           # CRITICAL, HIGH, MEDIUM, LOW
    "attack_type": "SQL_INJECTION",
    "confidence": 0.95,
    "model": "payload_cnn",
    "source": {
        "ip": "192.168.1.100",
        "payload": "' OR 1=1--",
        "url": "https://example.com/login"
    },
    "recommended_action": "Block IP and investigate",
    "explanation": {
        "top_features": ["single_quote", "OR_keyword", "comment_sequence"],
        "similar_attacks": ["SQL injection attempt"]
    }
}
```

### Configuration: config/alerts.yaml

```yaml
# Alert System Configuration

channels:
  webhook:
    enabled: true
    url: "${WEBHOOK_URL}"  # Set via environment variable
    format: "slack"        # slack, discord, teams
    retry_attempts: 3
    timeout_seconds: 10
  
  email:
    enabled: false
    smtp_host: "smtp.gmail.com"
    smtp_port: 587
    use_tls: true
    username: "${SMTP_USER}"
    password: "${SMTP_PASS}"
    from_address: "alerts@yourdomain.com"
    recipients:
      - "security@company.com"
      - "oncall@company.com"
  
  console:
    enabled: true
    colored: true
    log_file: "logs/alerts.log"

# Severity -> Channel routing
routing:
  CRITICAL: [webhook, email, console]
  HIGH: [webhook, console]
  MEDIUM: [console]
  LOW: [console]

# Rate limiting to prevent alert fatigue
rate_limit:
  enabled: true
  max_per_minute: 10
  aggregate_similar: true
  aggregation_window_seconds: 60

# Alert templates
templates:
  title: "🚨 {severity} Alert: {attack_type} Detected"
  body: |
    **Attack Type:** {attack_type}
    **Confidence:** {confidence:.1%}
    **Source:** {source}
    **Time:** {timestamp}
    **Action:** {recommended_action}
```

### Integration with Predictor

```python
# In batch_predictor.py - add alert integration

from src.alerts import AlertDispatcher

class BatchHybridPredictor:
    def __init__(self, ..., alert_dispatcher: AlertDispatcher = None):
        self.alert_dispatcher = alert_dispatcher
    
    def predict_batch(self, data, alert_threshold: float = 0.8):
        result = self._predict(data)
        
        # Send alerts for high-confidence attacks
        if self.alert_dispatcher and result['confidence'] > alert_threshold:
            self.alert_dispatcher.send({
                "severity": self._get_severity(result['confidence']),
                "attack_type": result.get('attack_type', 'UNKNOWN'),
                "confidence": result['confidence'],
                "source": data
            })
        
        return result
```

---

## Feature 2: A/B Testing Framework

### Purpose
Compare different model versions in production to determine which performs better using statistical testing.

### Existing Code to Leverage

**`src/model_registry.py`** provides:
- `ModelRegistry.save()` - Save model with version
- `ModelRegistry.load()` - Load specific version
- `ModelRegistry.list_versions()` - Get all versions
- `ModelRegistry.set_active()` - Set active version (rollback)
- `ModelRegistry.get_metrics()` - Get saved metrics

**`src/model_monitor.py`** provides:
- `ModelMonitor.log_prediction()` - Log latency/confidence
- `ModelMonitor.get_stats()` - Get aggregated stats
- `DriftDetector` - Statistical drift detection

### Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `src/ab_testing.py` | 200 | ABTestManager class |
| `config/ab_tests.yaml` | 50 | Experiment definitions |
| `tests/test_ab_testing.py` | 100 | Unit tests |

### Class: ABTestManager

**Location:** `src/ab_testing.py`

**Key Integration Points:**
- Uses `ModelRegistry` for loading versioned models
- Uses `ModelMonitor` for tracking per-variant metrics
- Stores experiment state in `evaluation/ab_tests/` directory

```python
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from scipy import stats
import numpy as np

from src.model_registry import ModelRegistry
from src.model_monitor import ModelMonitor


class ABTestManager:
    """Manage A/B testing experiments for model comparison."""
    
    def __init__(self, registry: ModelRegistry, monitor: ModelMonitor = None,
                 config_path: str = "config/ab_tests.yaml",
                 storage_path: str = "evaluation/ab_tests"):
        """
        Args:
            registry: ModelRegistry for loading model versions
            monitor: ModelMonitor for tracking metrics (optional)
            config_path: Path to experiment configuration
            storage_path: Directory for experiment data
        """
        self.registry = registry
        self.monitor = monitor or ModelMonitor()
        self.config = self._load_config(config_path)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, dict] = {}
        self._load_experiments()
    
    def create_experiment(self, name: str, model_a: dict, model_b: dict,
                         split_ratio: float = 0.5, min_samples: int = 1000,
                         max_duration_hours: int = 168) -> dict:
        """
        Create a new A/B test experiment.
        
        Args:
            name: Unique experiment name (e.g., "payload_v2_test")
            model_a: {"name": "payload_cnn", "version": "1.0.0"}
            model_b: {"name": "payload_cnn", "version": "2.0.0"}
            split_ratio: Fraction of traffic to model B (0.0-1.0)
            min_samples: Minimum samples before determining winner
            max_duration_hours: Auto-end after this duration
        
        Returns:
            Experiment configuration dict
        """
    
    def get_model(self, experiment_name: str, model_class=None) -> Tuple[any, str]:
        """
        Get model for prediction based on split ratio.
        
        Args:
            experiment_name: Name of active experiment
            model_class: PyTorch model class (required for .pt models)
        
        Returns:
            (model, variant_name) - variant_name is "A" or "B"
        
        Example:
            model, variant = ab_manager.get_model("payload_v2_test", PayloadCNN)
            result = model(input_tensor)
            ab_manager.record_outcome("payload_v2_test", variant, result)
        """
    
    def record_outcome(self, experiment_name: str, variant: str,
                       prediction: float, actual: int = None,
                       latency_ms: float = None):
        """
        Record prediction outcome for analysis.
        
        Args:
            experiment_name: Name of experiment
            variant: "A" or "B"
            prediction: Model's prediction (0-1 probability)
            actual: Ground truth if available (0 or 1)
            latency_ms: Prediction latency in milliseconds
        """
    
    def get_results(self, experiment_name: str) -> dict:
        """
        Get current results for an experiment.
        
        Returns:
            {
                "experiment": "payload_v2_test",
                "status": "running",  # running, completed, cancelled
                "duration_hours": 24.5,
                "samples": {"A": 523, "B": 477},
                "metrics": {
                    "A": {"accuracy": 0.94, "latency_p50": 12.3, "latency_p95": 45.2},
                    "B": {"accuracy": 0.96, "latency_p50": 11.8, "latency_p95": 42.1}
                },
                "winner": None,
                "p_value": 0.08,
                "significant": False,
                "recommendation": "Continue collecting data (need 477 more samples)"
            }
        """
    
    def determine_winner(self, experiment_name: str) -> dict:
        """
        Determine if there's a statistically significant winner.
        
        Uses:
        - Chi-squared test for accuracy comparison
        - Welch's t-test for latency comparison
        - Requires p-value < significance_level (default 0.05)
        
        Returns:
            {
                "winner": "B",  # or "A" or None
                "confidence": 0.95,
                "improvement": {"accuracy": 0.02, "latency": -0.5},
                "p_values": {"accuracy": 0.03, "latency": 0.12},
                "recommendation": "Deploy model B - 2% accuracy improvement"
            }
        """
    
    def end_experiment(self, experiment_name: str, deploy_winner: bool = False) -> dict:
        """
        End experiment and optionally deploy winner.
        
        Args:
            experiment_name: Name of experiment
            deploy_winner: If True, call registry.set_active() for winner
        
        Returns:
            Final experiment results with deployment status
        """
    
    def list_experiments(self, status: str = None) -> List[dict]:
        """List all experiments, optionally filtered by status."""
    
    def _calculate_significance(self, results_a: dict, results_b: dict) -> dict:
        """Calculate statistical significance using scipy.stats."""
    
    def _save_experiment(self, name: str):
        """Persist experiment state to JSON file."""
    
    def _load_experiments(self):
        """Load all experiment states from storage."""
```

### Experiment Data Structure

```python
{
    "name": "payload_v2_test",
    "status": "running",  # running, completed, cancelled
    "created_at": "2025-12-21T00:00:00Z",
    "ended_at": null,
    "model_a": {
        "name": "payload_cnn",
        "version": "1.0.0"
    },
    "model_b": {
        "name": "payload_cnn", 
        "version": "2.0.0"
    },
    "split_ratio": 0.5,
    "min_samples": 1000,
    "max_duration_hours": 168,
    "results": {
        "A": {
            "samples": 523,
            "predictions": [...],
            "latencies": [...],
            "correct": 492  # if ground truth available
        },
        "B": {
            "samples": 477,
            "predictions": [...],
            "latencies": [...],
            "correct": 461
        }
    }
}
```

### Configuration: config/ab_tests.yaml

```yaml
# A/B Testing Configuration

experiments:
  payload_v2_test:
    enabled: true
    model_a:
      name: "payload_cnn"
      version: "1.0.0"
    model_b:
      name: "payload_cnn"
      version: "2.0.0"
    split_ratio: 0.5
    min_samples: 1000
    max_duration_hours: 168  # 1 week
    auto_deploy_winner: false
    metrics:
      - accuracy
      - latency_p50
      - latency_p95
      - false_positive_rate

settings:
  significance_level: 0.05
  min_improvement_threshold: 0.02  # 2% improvement required
  storage_path: "evaluation/ab_tests/"
```

### Integration with BatchHybridPredictor

```python
# Enhanced prediction pipeline with A/B testing
from src.ab_testing import ABTestManager
from src.model_registry import ModelRegistry
from src.model_monitor import ModelMonitor

registry = ModelRegistry('models')
monitor = ModelMonitor()
ab_manager = ABTestManager(registry, monitor)

# In prediction loop
def predict_with_ab_test(data, experiment_name="payload_v2_test"):
    import time
    
    # Get model based on experiment split
    model, variant = ab_manager.get_model(experiment_name, PayloadCNN)
    
    # Make prediction with timing
    start = time.perf_counter()
    result = model(data)
    latency_ms = (time.perf_counter() - start) * 1000
    
    # Record outcome for analysis
    ab_manager.record_outcome(
        experiment_name,
        variant=variant,
        prediction=float(torch.sigmoid(result)),
        latency_ms=latency_ms
    )
    
    return result

# Check results periodically
results = ab_manager.get_results("payload_v2_test")
if results['significant']:
    print(f"Winner: {results['winner']} with {results['improvement']:.1%} improvement")
```

---

## Feature 3: Automated Retraining Trigger

### Purpose
Automatically trigger model retraining when data drift, performance degradation, or scheduled conditions are met.

### Existing Code to Leverage

**`src/model_monitor.py`** provides:
- `DriftDetector` - KS test for distribution drift
- `ModelMonitor` - Runtime metrics collection

**`src/training/`** provides:
- `train_payload.py` - Payload CNN training
- `train_url.py` - URL CNN training  
- `train_timeseries.py` - LSTM training

**`scripts/validate_realworld.py`** provides:
- Validation test suite (42 test cases)
- Pass/fail reporting

### Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `src/retraining_trigger.py` | 220 | RetrainingTrigger class |
| `config/retraining.yaml` | 60 | Trigger configuration |
| `tests/test_retraining.py` | 80 | Unit tests |

### Class: RetrainingTrigger

**Location:** `src/retraining_trigger.py`

```python
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
import threading
import time

from src.model_registry import ModelRegistry
from src.model_monitor import ModelMonitor, DriftDetector


class RetrainingTrigger:
    """Monitor conditions and trigger model retraining automatically."""
    
    def __init__(self, config_path: str = "config/retraining.yaml",
                 registry: ModelRegistry = None,
                 monitor: ModelMonitor = None,
                 alert_dispatcher = None):
        """
        Initialize with dependencies.
        
        Args:
            config_path: Path to YAML configuration
            registry: ModelRegistry for version management
            monitor: ModelMonitor for performance tracking
            alert_dispatcher: AlertDispatcher for notifications
        """
        self.config = self._load_config(config_path)
        self.registry = registry or ModelRegistry('models')
        self.monitor = monitor or ModelMonitor()
        self.alert_dispatcher = alert_dispatcher
        self._drift_detectors: Dict[str, DriftDetector] = {}
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_daemon = False
    
    def check_conditions(self, model_name: str) -> dict:
        """
        Check all trigger conditions for a model.
        
        Returns:
            {
                "model": "payload_cnn",
                "should_retrain": True,
                "triggers_fired": ["drift", "schedule"],
                "details": {
                    "drift": {"detected": True, "p_value": 0.02, "statistic": 0.15},
                    "performance": {"current": 0.96, "threshold": 0.95, "triggered": False},
                    "schedule": {"due": True, "last_trained": "2025-12-14T00:00:00Z"},
                    "volume": {"new_samples": 5000, "threshold": 10000, "triggered": False}
                }
            }
        """
    
    def should_retrain(self, model_name: str) -> bool:
        """Simple boolean check if retraining needed."""
        return self.check_conditions(model_name)['should_retrain']
    
    def trigger_retraining(self, model_name: str, reason: str = None,
                           blocking: bool = True) -> dict:
        """
        Launch retraining job for a model.
        
        Args:
            model_name: Name of model to retrain
            reason: Reason for retraining (for logging)
            blocking: If True, wait for completion
        
        Returns:
            {
                "status": "completed",  # or "started", "failed"
                "job_id": "retrain-20251221-001",
                "model": "payload_cnn",
                "reason": "drift_detected",
                "duration_seconds": 342.5,
                "new_version": "2.1.0",
                "validation_passed": True
            }
        """
    
    def validate_new_model(self, model_name: str, model_path: str) -> dict:
        """
        Run validation tests on newly trained model.
        
        Uses scripts/validate_realworld.py test suite.
        
        Returns:
            {
                "passed": True,
                "metrics": {"accuracy": 0.97, "false_positive_rate": 0.01},
                "validation_tests": 42,
                "tests_passed": 40,
                "failures": [...]
            }
        """
    
    def deploy_if_better(self, model_name: str, new_model_path: str,
                         min_improvement: float = 0.02) -> dict:
        """
        Compare new model with current and deploy if better.
        
        Args:
            model_name: Name of model
            new_model_path: Path to newly trained model
            min_improvement: Minimum accuracy improvement required
        
        Returns:
            {
                "deployed": True,
                "old_version": "2.0.0",
                "new_version": "2.1.0",
                "improvement": 0.03,
                "reason": "3% accuracy improvement"
            }
        """
    
    def run_daemon(self, interval_minutes: int = 60):
        """
        Run as background daemon, checking conditions periodically.
        
        This starts a background thread - call stop_daemon() to stop.
        """
    
    def stop_daemon(self):
        """Stop the background daemon thread."""
    
    def _check_drift(self, model_name: str, recent_data: list) -> dict:
        """Check for data drift using DriftDetector."""
    
    def _check_performance(self, model_name: str) -> dict:
        """Check if performance below threshold using ModelMonitor."""
    
    def _check_schedule(self, model_name: str) -> dict:
        """Check if scheduled retraining is due (cron-like)."""
    
    def _check_volume(self, model_name: str) -> dict:
        """Check if new data volume exceeds threshold."""
    
    def _run_training_script(self, model_name: str) -> subprocess.CompletedProcess:
        """Execute the appropriate training script."""
    
    def _notify(self, event: str, details: dict):
        """Send notification via alert_dispatcher if configured."""
```

### Configuration: config/retraining.yaml

```yaml
# Automated Retraining Configuration

models:
  payload_cnn:
    training_script: "src/training/train_payload.py"
    triggers:
      drift:
        enabled: true
        threshold: 0.05  # p-value threshold
        check_features: ["char_distribution", "length_distribution"]
      performance:
        enabled: true
        min_accuracy: 0.95
        max_false_positive_rate: 0.05
        evaluation_window_hours: 24
      schedule:
        enabled: true
        cron: "0 2 * * 0"  # Every Sunday at 2 AM
      volume:
        enabled: false
        new_samples_threshold: 10000
        data_path: "datasets/new_samples/"
    validation:
      min_accuracy: 0.90
      max_false_positive_rate: 0.10
      test_samples: 1000
    auto_deploy: false  # Require manual approval
    notify_on_trigger: true

  url_cnn:
    training_script: "src/training/train_url.py"
    triggers:
      drift:
        enabled: true
        threshold: 0.05
      schedule:
        enabled: true
        cron: "0 3 * * 0"  # Every Sunday at 3 AM
    auto_deploy: true
    notify_on_trigger: true

  timeseries_lstm:
    training_script: "src/training/train_timeseries.py"
    triggers:
      schedule:
        enabled: true
        cron: "0 4 * * 0"
    auto_deploy: true

daemon:
  enabled: true
  check_interval_minutes: 60
  log_path: "logs/retraining.log"

notifications:
  on_trigger: true
  on_success: true
  on_failure: true
  channels: [webhook, console]
```

### Retraining Pipeline Flow

```
┌─────────────────┐
│ Check Conditions│
│ (drift/perf/    │
│  schedule/vol)  │
└────────┬────────┘
         │ trigger fired
         ▼
┌─────────────────┐
│ Send Alert      │
│ "Retraining     │
│  Started"       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Run Training    │
│ Script          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validate New    │
│ Model           │
└────────┬────────┘
         │
    ┌────┴────┐
    │ Passed? │
    └────┬────┘
    Yes  │  No
    ▼    │   ▼
┌───────┐│┌───────┐
│Compare││ │Alert  │
│with   │││Failure│
│Current│││       │
└───┬───┘│└───────┘
    │    │
    ▼    │
┌───────┐│
│Better?││
└───┬───┘│
Yes │ No │
    ▼    ▼
┌───────┐┌───────┐
│Deploy ││Keep   │
│New    ││Current│
└───────┘└───────┘
```

---


## Feature 4: Gradio Dashboard

### Purpose
Interactive web interface for testing predictions, viewing explanations, and monitoring model performance.

### Existing Code to Leverage

**`src/hybrid_predictor.py`** provides:
- `HybridPredictor.predict_payload()` - Payload prediction
- `HybridPredictor.predict_url()` - URL prediction
- `HybridPredictor.predict()` - Full ensemble prediction

**`src/explainability_v2.py`** provides:
- `PayloadExplainer` - Character-level importance
- `URLExplainer` - Feature-based explanation
- `generate_explanation()` - Human-readable explanations

**`src/input_validator.py`** provides:
- `InputValidator.validate_payload()` - Sanitize payloads
- `InputValidator.validate_url()` - Sanitize URLs

### Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 280 | Main Gradio application |
| `src/dashboard/__init__.py` | 5 | Package init |
| `src/dashboard/explainer.py` | 100 | Enhanced explainer wrapper |

### Gradio App Structure

**Location:** `app.py` (root level for HF Spaces)

```python
import gradio as gr
import torch
import numpy as np
from pathlib import Path

# Import existing components
from src.hybrid_predictor import HybridPredictor
from src.input_validator import InputValidator
from src.explainability_v2 import generate_explanation
from src.dashboard.explainer import ModelExplainer

# Global predictor instance
predictor = None
explainer = None

def load_models():
    """Load models on startup."""
    global predictor, explainer
    predictor = HybridPredictor('models')
    predictor.load_models()
    if 'payload_cnn' in predictor.pytorch_models:
        explainer = ModelExplainer(predictor.pytorch_models['payload_cnn'])

def scan_payload(text: str) -> tuple:
    """
    Analyze text payload for attacks.
    
    Returns:
        (result_dict, explanation_plot)
    """
    if not text.strip():
        return {"error": "Please enter a payload"}, None
    
    # Validate input
    validator = InputValidator()
    try:
        text = validator.validate_payload(text)
    except Exception as e:
        return {"error": str(e)}, None
    
    # Get prediction
    result = predictor.predict({'payloads': [text]})
    confidence = float(result['confidence'][0])
    is_attack = bool(result['is_attack'][0])
    
    # Generate explanation
    explanation = generate_explanation(result, {'payloads': [text]})
    
    # Create explanation plot if explainer available
    plot = None
    if explainer:
        exp_result = explainer.explain_payload(text)
        plot = exp_result.get('shap_plot')
    
    output = {
        "is_attack": is_attack,
        "confidence": f"{confidence:.1%}",
        "severity": explanation['verdict'],
        "attack_type": _classify_attack_type(text, confidence),
        "explanation": explanation['reasons'],
        "component_scores": explanation['component_scores']
    }
    
    return output, plot

def scan_url(url: str) -> dict:
    """Analyze URL for maliciousness."""
    if not url.strip():
        return {"error": "Please enter a URL"}
    
    validator = InputValidator()
    try:
        url = validator.validate_url(url)
    except Exception as e:
        return {"error": str(e)}
    
    result = predictor.predict({'urls': [url]})
    confidence = float(result['confidence'][0])
    
    return {
        "is_malicious": bool(result['is_attack'][0]),
        "confidence": f"{confidence:.1%}",
        "url_analyzed": url,
        "recommendation": "Block" if confidence > 0.7 else "Allow"
    }

def batch_scan(file) -> tuple:
    """Process uploaded file with multiple inputs."""
    if file is None:
        return None, None
    
    # Read file
    content = Path(file.name).read_text()
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    
    results = []
    for line in lines[:100]:  # Limit to 100
        result = predictor.predict({'payloads': [line]})
        results.append({
            'input': line[:50] + '...' if len(line) > 50 else line,
            'is_attack': bool(result['is_attack'][0]),
            'confidence': f"{float(result['confidence'][0]):.1%}"
        })
    
    # Create downloadable CSV
    import csv
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=['input', 'is_attack', 'confidence'])
        writer.writeheader()
        writer.writerows(results)
        output_path = f.name
    
    return results, output_path

def get_model_info() -> dict:
    """Return model performance metrics."""
    return {
        "models_loaded": list(predictor.pytorch_models.keys()) + list(predictor.sklearn_models.keys()),
        "device": str(predictor.device),
        "validation_accuracy": {
            "payload_cnn": "99.89%",
            "url_cnn": "97.47%",
            "timeseries_lstm": "75.38%"
        },
        "known_limitations": [
            "<3 emoji may trigger false positive",
            "SELECT FROM menu flagged as suspicious",
            "Emails with dots borderline"
        ]
    }

def _classify_attack_type(text: str, confidence: float) -> str:
    """Classify attack type based on patterns."""
    text_lower = text.lower()
    if any(p in text_lower for p in ["'", "or ", "union", "select", "--"]):
        return "SQL_INJECTION"
    if any(p in text_lower for p in ["<script", "onerror", "onload", "javascript:"]):
        return "XSS"
    if any(p in text_lower for p in [";", "|", "`", "$("]):
        return "COMMAND_INJECTION"
    if any(p in text_lower for p in ["{{", "${", "<%"]):
        return "TEMPLATE_INJECTION"
    return "UNKNOWN" if confidence > 0.5 else "BENIGN"

# Build Gradio Interface
with gr.Blocks(title="AI Hacking Detection", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🛡️ AI Hacking Detection System")
    gr.Markdown("Detect SQL injection, XSS, command injection, and malicious URLs")
    
    with gr.Tabs():
        # Tab 1: Payload Scanner
        with gr.TabItem("🔍 Payload Scanner"):
            payload_input = gr.Textbox(
                label="Enter payload to analyze",
                placeholder="e.g., ' OR 1=1--",
                lines=3
            )
            payload_btn = gr.Button("Analyze", variant="primary")
            payload_output = gr.JSON(label="Analysis Result")
            payload_explanation = gr.Plot(label="Feature Importance")
        
        # Tab 2: URL Scanner
        with gr.TabItem("🌐 URL Scanner"):
            url_input = gr.Textbox(
                label="Enter URL to analyze",
                placeholder="https://example.com"
            )
            url_btn = gr.Button("Analyze", variant="primary")
            url_output = gr.JSON(label="Analysis Result")
        
        # Tab 3: Batch Analysis
        with gr.TabItem("📁 Batch Analysis"):
            file_input = gr.File(label="Upload CSV/TXT file")
            batch_btn = gr.Button("Process", variant="primary")
            batch_output = gr.Dataframe(label="Results")
            download_btn = gr.File(label="Download Results")
        
        # Tab 4: Model Performance
        with gr.TabItem("📊 Model Info"):
            gr.Markdown("## Model Performance")
            metrics_display = gr.JSON(label="Current Metrics")
            gr.Markdown("## Known Limitations")
            gr.Dataframe(
                value=[
                    ["<3 emoji", "False positive (~95%)", "<character resembles HTML tag"],
                    ["SELECT FROM menu", "Flagged (~72%)", "Ambiguous SQL-like pattern"],
                    ["Emails with dots", "Borderline (~52%)", "Dot patterns in payloads"]
                ],
                headers=["Pattern", "Behavior", "Reason"]
            )
        
        # Tab 5: API Documentation
        with gr.TabItem("📖 API Docs"):
            gr.Markdown("""
            ## REST API Usage
            
            ### Endpoint: POST /api/v1/predict/payload
            ```bash
            curl -X POST https://your-space.hf.space/api/v1/predict/payload \\
              -H "Content-Type: application/json" \\
              -d '{"payload": "' OR 1=1--"}'
            ```
            
            ### Response
            ```json
            {
              "is_attack": true,
              "confidence": 0.95,
              "attack_type": "SQL_INJECTION",
              "severity": "HIGH"
            }
            ```
            """)
        
        # Tab 6: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## AI Hacking Detection System
            
            **Models:**
            - Payload CNN: 99.89% accuracy
            - URL CNN: 97.47% accuracy
            - Time-Series LSTM: 75.38% accuracy
            
            **Detects:**
            - SQL Injection
            - Cross-Site Scripting (XSS)
            - Command Injection
            - Template Injection
            - Malicious URLs
            - Phishing Domains
            
            **License:** MIT
            
            **GitHub:** [Repository Link]
            """)
    
    # Event handlers
    payload_btn.click(scan_payload, inputs=payload_input, outputs=[payload_output, payload_explanation])
    url_btn.click(scan_url, inputs=url_input, outputs=url_output)
    batch_btn.click(batch_scan, inputs=file_input, outputs=[batch_output, download_btn])

demo.launch()
```

### Class: ModelExplainer

**Location:** `src/dashboard/explainer.py`

```python
import shap
import numpy as np
import plotly.graph_objects as go

class ModelExplainer:
    """Generate SHAP explanations for model predictions."""
    
    def __init__(self, model, model_type: str = "payload"):
        """
        Args:
            model: Trained model
            model_type: "payload", "url", or "network"
        """
        self.model = model
        self.model_type = model_type
        self.explainer = None
    
    def explain_payload(self, text: str) -> dict:
        """
        Generate explanation for payload prediction.
        
        Returns:
            {
                "prediction": 0.95,
                "is_attack": True,
                "top_features": [
                    {"feature": "single_quote", "importance": 0.3},
                    {"feature": "OR_keyword", "importance": 0.25},
                    {"feature": "comment_chars", "importance": 0.2}
                ],
                "explanation_text": "High confidence SQL injection due to...",
                "shap_plot": <plotly figure>
            }
        """
    
    def explain_url(self, url: str) -> dict:
        """Generate explanation for URL prediction."""
    
    def get_feature_importance(self) -> dict:
        """Get global feature importance for model."""
    
    def create_waterfall_plot(self, shap_values, feature_names) -> go.Figure:
        """Create SHAP waterfall plot using Plotly."""
    
    def _tokenize_for_explanation(self, text: str) -> list:
        """Convert text to explainable tokens."""
    
    def generate_text_explanation(self, prediction: float, 
                                   top_features: list) -> str:
        """Generate human-readable explanation."""
```

### Explanation Output Example

```python
{
    "prediction": 0.95,
    "is_attack": True,
    "attack_type": "SQL_INJECTION",
    "severity": "HIGH",
    "explanation": {
        "summary": "This input is likely a SQL injection attack (95% confidence)",
        "top_features": [
            {"feature": "' (single quote)", "importance": 0.30, "direction": "attack"},
            {"feature": "OR keyword", "importance": 0.25, "direction": "attack"},
            {"feature": "-- (comment)", "importance": 0.20, "direction": "attack"},
            {"feature": "= (equals)", "importance": 0.15, "direction": "attack"}
        ],
        "similar_patterns": [
            "Classic OR-based SQL injection",
            "Authentication bypass attempt"
        ],
        "recommendation": "Block this request and investigate source IP"
    }
}
```

---

## Feature 5: API Server

### Purpose
Production-ready REST API for integration with other systems, with validation, monitoring, and optional alerting.

### Existing Code to Leverage

**`src/batch_predictor.py`** provides:
- `BatchHybridPredictor` - Optimized batch predictions
- Built-in validation via `InputValidator`
- Built-in monitoring via `ModelMonitor`

**`src/input_validator.py`** provides:
- Input sanitization and validation
- `ValidationError` for invalid inputs

### Files to Create

| File | Lines | Purpose |
|------|-------|---------|
| `src/api/__init__.py` | 10 | Package exports |
| `src/api/server.py` | 70 | FastAPI application |
| `src/api/routes/__init__.py` | 5 | Routes package |
| `src/api/routes/predict.py` | 90 | Prediction endpoints |
| `src/api/routes/health.py` | 35 | Health check endpoints |
| `src/api/schemas.py` | 60 | Pydantic models |
| `src/api/middleware.py` | 50 | Rate limiting, logging |
| `tests/test_api.py` | 120 | API tests |

### FastAPI Server

**Location:** `src/api/server.py`

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

from src.batch_predictor import BatchHybridPredictor
from src.model_monitor import ModelMonitor
from src.api.routes import predict, health
from src.api.middleware import RateLimitMiddleware, LoggingMiddleware

# Global state
predictor = None
monitor = None
start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global predictor, monitor, start_time
    
    # Startup
    print("Loading models...")
    monitor = ModelMonitor()
    predictor = BatchHybridPredictor(
        models_dir='models',
        validator=True,
        monitor=monitor
    )
    predictor.load_models()
    start_time = time.time()
    print(f"Loaded {len(predictor.pytorch_models)} PyTorch, {len(predictor.sklearn_models)} sklearn models")
    
    yield
    
    # Shutdown
    print("Shutting down...")

app = FastAPI(
    title="AI Hacking Detection API",
    description="Real-time cyber attack detection using ensemble ML models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=100)

# Include routers
app.include_router(predict.router, prefix="/api/v1")
app.include_router(health.router)

def get_predictor() -> BatchHybridPredictor:
    """Dependency injection for predictor."""
    return predictor

def get_monitor() -> ModelMonitor:
    """Dependency injection for monitor."""
    return monitor

def get_uptime() -> float:
    """Get server uptime in seconds."""
    return time.time() - start_time if start_time else 0
```

### Prediction Routes

**Location:** `src/api/routes/predict.py`

```python
from fastapi import APIRouter, HTTPException, Depends
from typing import List
import time

from src.api.schemas import (
    PayloadRequest, URLRequest, BatchRequest,
    PredictResponse, BatchResponse
)
from src.api.server import get_predictor, get_monitor
from src.input_validator import ValidationError
from src.alerts import AlertDispatcher

router = APIRouter(prefix="/predict", tags=["Prediction"])

# Optional alert dispatcher
alert_dispatcher = None
try:
    alert_dispatcher = AlertDispatcher()
except:
    pass

@router.post("/payload", response_model=PredictResponse)
async def predict_payload(request: PayloadRequest, predictor=Depends(get_predictor)):
    """
    Analyze payload for attacks.
    
    - **payload**: Text to analyze (max 10,000 chars)
    - **include_explanation**: Include feature importance explanation
    
    Returns prediction with confidence score and severity level.
    """
    start = time.perf_counter()
    
    try:
        result = predictor.predict_batch({'payloads': [request.payload]})
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    confidence = float(result['confidence'][0])
    is_attack = bool(result['is_attack'][0])
    
    # Send alert if high confidence attack
    if alert_dispatcher and confidence > 0.8 and is_attack:
        alert_dispatcher.send_from_prediction(result, {'payload': request.payload})
    
    response = PredictResponse(
        is_attack=is_attack,
        confidence=confidence,
        attack_type=_classify_attack(request.payload) if is_attack else None,
        severity=_get_severity(confidence),
        processing_time_ms=(time.perf_counter() - start) * 1000
    )
    
    if request.include_explanation:
        response.explanation = _generate_explanation(result, request.payload)
    
    return response

@router.post("/url", response_model=PredictResponse)
async def predict_url(request: URLRequest, predictor=Depends(get_predictor)):
    """
    Analyze URL for maliciousness.
    
    - **url**: URL to analyze (max 2,000 chars)
    - **include_explanation**: Include feature importance explanation
    """
    start = time.perf_counter()
    
    try:
        result = predictor.predict_batch({'urls': [request.url]})
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    confidence = float(result['confidence'][0])
    
    return PredictResponse(
        is_attack=bool(result['is_attack'][0]),
        confidence=confidence,
        attack_type="MALICIOUS_URL" if confidence > 0.5 else None,
        severity=_get_severity(confidence),
        processing_time_ms=(time.perf_counter() - start) * 1000
    )

@router.post("/batch", response_model=BatchResponse)
async def predict_batch(request: BatchRequest, predictor=Depends(get_predictor)):
    """
    Batch prediction for multiple inputs.
    
    - **payloads**: List of payloads (max 100)
    - **urls**: List of URLs (max 100)
    """
    start = time.perf_counter()
    
    data = {}
    if request.payloads:
        data['payloads'] = request.payloads[:100]
    if request.urls:
        data['urls'] = request.urls[:100]
    
    if not data:
        raise HTTPException(status_code=422, detail="No inputs provided")
    
    try:
        result = predictor.predict_batch(data)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    return BatchResponse(
        results=[
            PredictResponse(
                is_attack=bool(result['is_attack'][i]),
                confidence=float(result['confidence'][i]),
                severity=_get_severity(float(result['confidence'][i])),
                processing_time_ms=0
            )
            for i in range(len(result['is_attack']))
        ],
        total_processing_time_ms=(time.perf_counter() - start) * 1000
    )

def _get_severity(confidence: float) -> str:
    if confidence > 0.95: return "CRITICAL"
    if confidence > 0.85: return "HIGH"
    if confidence > 0.7: return "MEDIUM"
    return "LOW"

def _classify_attack(text: str) -> str:
    text_lower = text.lower()
    if any(p in text_lower for p in ["'", "union", "select"]): return "SQL_INJECTION"
    if any(p in text_lower for p in ["<script", "onerror"]): return "XSS"
    if any(p in text_lower for p in [";", "|", "`"]): return "COMMAND_INJECTION"
    return "UNKNOWN"

def _generate_explanation(result: dict, text: str) -> dict:
    return {
        "summary": f"Detected potential attack with {result['confidence'][0]:.1%} confidence",
        "component_scores": {k: float(v[0]) for k, v in result['scores'].items()}
    }
```

### Health Routes

**Location:** `src/api/routes/health.py`

```python
from fastapi import APIRouter, Depends
from src.api.schemas import HealthResponse, ReadinessResponse
from src.api.server import get_predictor, get_monitor, get_uptime

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness check - returns 200 if server is running."""
    return HealthResponse(status="healthy", uptime_seconds=get_uptime())

@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(predictor=Depends(get_predictor), monitor=Depends(get_monitor)):
    """Readiness check - returns 200 if models are loaded."""
    models_loaded = {
        "pytorch": list(predictor.pytorch_models.keys()) if predictor else [],
        "sklearn": list(predictor.sklearn_models.keys()) if predictor else []
    }
    
    is_ready = len(models_loaded["pytorch"]) > 0 or len(models_loaded["sklearn"]) > 0
    
    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        models_loaded=models_loaded,
        uptime_seconds=get_uptime(),
        metrics=monitor.get_stats() if monitor else {}
    )
```

### Request/Response Schemas

**Location:** `src/api/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PayloadRequest(BaseModel):
    payload: str = Field(..., max_length=10000, description="Text payload to analyze")
    include_explanation: bool = Field(False, description="Include feature explanation")

class URLRequest(BaseModel):
    url: str = Field(..., max_length=2000, description="URL to analyze")
    include_explanation: bool = Field(False, description="Include feature explanation")

class BatchRequest(BaseModel):
    payloads: Optional[List[str]] = Field(None, max_length=100)
    urls: Optional[List[str]] = Field(None, max_length=100)

class PredictResponse(BaseModel):
    is_attack: bool
    confidence: float = Field(..., ge=0, le=1)
    attack_type: Optional[str] = None
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    explanation: Optional[Dict[str, Any]] = None
    processing_time_ms: float

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    total_processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float

class ReadinessResponse(BaseModel):
    status: str
    models_loaded: Dict[str, List[str]]
    uptime_seconds: float
    metrics: Dict[str, Any] = {}
```

### API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict/payload` | POST | Analyze payload |
| `/api/v1/predict/url` | POST | Analyze URL |
| `/api/v1/predict/batch` | POST | Batch prediction |
| `/health` | GET | Liveness check |
| `/health/ready` | GET | Readiness (models loaded) |
| `/docs` | GET | Swagger UI |

---

## Integration Map

### Component Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                     PHASE B INTEGRATION MAP                      │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   config/*.yaml   │
                    │  (alerts, ab,    │
                    │   retraining)    │
                    └────────┬─────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  AlertDispatcher │ │  ABTestManager  │ │RetrainingTrigger│
│  (src/alerts/)   │ │(src/ab_testing) │ │(src/retraining) │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         │    ┌──────────────┼──────────────┐    │
         │    │              │              │    │
         ▼    ▼              ▼              ▼    ▼
    ┌─────────────────────────────────────────────────┐
    │              EXISTING INFRASTRUCTURE             │
    ├─────────────────────────────────────────────────┤
    │  ModelRegistry    ModelMonitor    DriftDetector │
    │  (versioning)     (metrics)       (drift)       │
    └────────────────────────┬────────────────────────┘
                             │
                             ▼
    ┌─────────────────────────────────────────────────┐
    │            BatchHybridPredictor                  │
    │  (src/batch_predictor.py)                       │
    │  - Payload CNN, URL CNN, LSTM                   │
    │  - InputValidator integration                   │
    │  - ModelMonitor integration                     │
    └────────────────────────┬────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
    │  API Server │  │  Dashboard  │  │ Explainers  │
    │ (FastAPI)   │  │  (Gradio)   │  │ (SHAP)      │
    └─────────────┘  └─────────────┘  └─────────────┘
```

### Data Flow

```
User Request → API/Dashboard → InputValidator → BatchHybridPredictor
                                                        │
                    ┌───────────────────────────────────┤
                    │                                   │
                    ▼                                   ▼
            ModelMonitor                         AlertDispatcher
            (log metrics)                        (if confidence > threshold)
                    │                                   │
                    ▼                                   ▼
            ABTestManager                        Notification Channels
            (if experiment active)               (webhook/email/console)
```

### File-to-File Dependencies

| New File | Depends On (Existing) | Depends On (New) |
|----------|----------------------|------------------|
| `src/alerts/dispatcher.py` | `src/alert_manager.py` | `src/alerts/channels/*.py` |
| `src/ab_testing.py` | `src/model_registry.py`, `src/model_monitor.py` | - |
| `src/retraining_trigger.py` | `src/model_monitor.py`, `src/training/*.py` | `src/alerts/dispatcher.py` |
| `app.py` | `src/hybrid_predictor.py`, `src/explainability_v2.py` | - |
| `src/api/server.py` | `src/batch_predictor.py`, `src/input_validator.py` | `src/api/routes/*.py` |

---

## Implementation Order

| Phase | Feature | Files | Est. Lines | Priority | Dependencies |
|-------|---------|-------|------------|----------|--------------|
| B1 | Alert System | 9 | 320 | High | None |
| B2 | API Server | 7 | 300 | High | Alert System (optional) |
| B3 | Gradio Dashboard | 3 | 350 | High | None |
| B4 | A/B Testing | 3 | 250 | Medium | ModelRegistry, ModelMonitor |
| B5 | Retraining Trigger | 3 | 280 | Medium | DriftDetector, Alert System |

**Total: ~1,500 lines across 25 files**

### Recommended Implementation Sequence

1. **Alert System** (B1) - Foundation for all notifications
2. **API Server** (B2) - Enables external integration, uses alerts
3. **Gradio Dashboard** (B3) - User-facing interface
4. **A/B Testing** (B4) - Requires stable prediction pipeline
5. **Retraining Trigger** (B5) - Requires stable monitoring + alerts

---

## File Summary

| Category | Files | Lines | New Tests |
|----------|-------|-------|-----------|
| Alert System | 9 | 320 | `tests/test_alerts.py` (80) |
| A/B Testing | 3 | 250 | `tests/test_ab_testing.py` (100) |
| Retraining | 3 | 280 | `tests/test_retraining.py` (80) |
| API Server | 7 | 300 | `tests/test_api.py` (120) |
| Dashboard | 3 | 350 | Manual testing |
| **Total** | **25** | **~1,500** | **~380** |

### Complete File List

```
# Alert System (B1)
src/alerts/__init__.py              # 10 lines
src/alerts/dispatcher.py            # 100 lines
src/alerts/channels/__init__.py     # 5 lines
src/alerts/channels/base.py         # 30 lines
src/alerts/channels/webhook.py      # 55 lines
src/alerts/channels/email.py        # 60 lines
src/alerts/channels/console.py      # 25 lines
config/alerts.yaml                  # 45 lines
tests/test_alerts.py                # 80 lines

# A/B Testing (B4)
src/ab_testing.py                   # 200 lines
config/ab_tests.yaml                # 50 lines
tests/test_ab_testing.py            # 100 lines

# Retraining Trigger (B5)
src/retraining_trigger.py           # 220 lines
config/retraining.yaml              # 60 lines
tests/test_retraining.py            # 80 lines

# API Server (B2)
src/api/__init__.py                 # 10 lines
src/api/server.py                   # 70 lines
src/api/routes/__init__.py          # 5 lines
src/api/routes/predict.py           # 90 lines
src/api/routes/health.py            # 35 lines
src/api/schemas.py                  # 60 lines
src/api/middleware.py               # 50 lines
tests/test_api.py                   # 120 lines

# Dashboard (B3)
app.py                              # 280 lines
src/dashboard/__init__.py           # 5 lines
src/dashboard/explainer.py          # 100 lines
```

---

## Requirements Additions

Add to `requirements.txt`:

```
# API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0
python-multipart>=0.0.6

# Dashboard
gradio>=4.44.0
plotly>=5.18.0

# Config & Utils
pyyaml>=6.0
python-dotenv>=1.0.0

# Explainability
shap>=0.43.0

# Statistical Testing (for A/B)
scipy>=1.11.0

# Scheduling (for retraining)
croniter>=2.0.0

# Monitoring (optional)
prometheus-client>=0.19.0
```

---

## Testing Strategy

### Unit Tests

Each new module should have corresponding tests:

```python
# tests/test_alerts.py
class TestAlertDispatcher:
    def test_send_to_webhook(self): ...
    def test_send_to_email(self): ...
    def test_rate_limiting(self): ...
    def test_severity_routing(self): ...
    def test_alert_aggregation(self): ...

# tests/test_ab_testing.py
class TestABTestManager:
    def test_create_experiment(self): ...
    def test_traffic_split(self): ...
    def test_record_outcome(self): ...
    def test_statistical_significance(self): ...
    def test_determine_winner(self): ...

# tests/test_retraining.py
class TestRetrainingTrigger:
    def test_drift_detection_trigger(self): ...
    def test_schedule_trigger(self): ...
    def test_performance_trigger(self): ...
    def test_validation_gate(self): ...
    def test_deploy_if_better(self): ...

# tests/test_api.py
class TestPredictionAPI:
    def test_payload_endpoint(self): ...
    def test_url_endpoint(self): ...
    def test_batch_endpoint(self): ...
    def test_health_check(self): ...
    def test_invalid_input(self): ...
```

### Integration Tests

```python
# tests/test_integration.py
class TestEndToEnd:
    def test_prediction_with_alert(self):
        """Prediction triggers alert when confidence > threshold."""
    
    def test_ab_test_with_monitoring(self):
        """A/B test records metrics to ModelMonitor."""
    
    def test_retraining_pipeline(self):
        """Full retraining: trigger → train → validate → deploy."""
```

### Manual Testing Checklist

#### Alert System
- [ ] Webhook sends to Slack/Discord (test with webhook.site)
- [ ] Email sends via SMTP (test with mailtrap.io)
- [ ] Rate limiting prevents alert flood
- [ ] Severity routing sends to correct channels
- [ ] Alert aggregation groups similar alerts

#### A/B Testing
- [ ] Traffic splits approximately 50/50
- [ ] Metrics recorded per variant
- [ ] Statistical test returns valid p-value
- [ ] Winner determination correct
- [ ] Experiment persists across restarts

#### Retraining Trigger
- [ ] Drift detection triggers retraining
- [ ] Schedule triggers at correct time
- [ ] Validation gate blocks bad models
- [ ] Deployment only if improvement > threshold
- [ ] Notifications sent on trigger/success/failure

#### Dashboard
- [ ] Payload scanner returns predictions
- [ ] URL scanner returns predictions
- [ ] Batch upload processes file
- [ ] Explanations display correctly
- [ ] Model info shows current metrics

#### API
- [ ] All endpoints respond with 200
- [ ] Health checks return correct status
- [ ] Invalid input returns 422
- [ ] CORS headers present
- [ ] Swagger UI accessible at /docs

---

## Risk Assessment

| Feature | Risk Level | Mitigation |
|---------|------------|------------|
| Alert System | Low | Clear interfaces, isolated channels |
| A/B Testing | Medium | Thorough statistical testing, fallback to weighted average |
| Retraining | Medium | Validation gate, manual approval option |
| Dashboard | Low | Wrapper around existing code |
| API | Low | Wrapper around existing code |

---

## Success Criteria

### Phase B Complete When:

1. **Alert System**: Alerts sent to webhook on high-confidence predictions
2. **A/B Testing**: Can run experiment comparing two model versions
3. **Retraining**: Drift detection triggers retraining with validation
4. **Dashboard**: Interactive Gradio UI runs locally with all tabs functional
5. **API**: REST endpoints accessible with Swagger documentation

### Performance Targets

| Metric | Target |
|--------|--------|
| API Latency (p95) | < 100ms |
| Dashboard Load Time | < 3s |
| Alert Delivery | < 5s |
| A/B Split Accuracy | ±5% of target |

---

*Last Updated: December 21, 2025*
