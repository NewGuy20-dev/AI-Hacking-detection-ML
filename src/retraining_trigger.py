"""Automated retraining trigger based on drift, performance, and schedule."""
import subprocess
import json
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional

from src.model_registry import ModelRegistry
from src.model_monitor import ModelMonitor, DriftDetector


class RetrainingTrigger:
    """Monitor conditions and trigger model retraining automatically."""
    
    TRAINING_SCRIPTS = {
        'payload_cnn': 'src/training/train_payload.py',
        'url_cnn': 'src/training/train_url.py',
        'timeseries_lstm': 'src/training/train_timeseries.py'
    }
    
    def __init__(self, registry: ModelRegistry = None, monitor: ModelMonitor = None,
                 alert_dispatcher=None, config: dict = None):
        self.registry = registry or ModelRegistry('models')
        self.monitor = monitor or ModelMonitor()
        self.alert_dispatcher = alert_dispatcher
        self.config = config or self._default_config()
        self._drift_detectors: Dict[str, DriftDetector] = {}
        self._last_trained: Dict[str, datetime] = {}
        self._daemon_thread: Optional[threading.Thread] = None
        self._stop_daemon = False
    
    def _default_config(self) -> dict:
        return {
            'drift_threshold': 0.05,
            'min_accuracy': 0.90,
            'schedule_days': 7,
            'auto_deploy': False,
            'min_improvement': 0.02
        }
    
    def check_conditions(self, model_name: str) -> dict:
        """Check all trigger conditions for a model."""
        triggers_fired = []
        details = {}
        
        # Check drift
        drift_result = self._check_drift(model_name)
        details['drift'] = drift_result
        if drift_result.get('triggered'):
            triggers_fired.append('drift')
        
        # Check performance
        perf_result = self._check_performance(model_name)
        details['performance'] = perf_result
        if perf_result.get('triggered'):
            triggers_fired.append('performance')
        
        # Check schedule
        schedule_result = self._check_schedule(model_name)
        details['schedule'] = schedule_result
        if schedule_result.get('triggered'):
            triggers_fired.append('schedule')
        
        return {
            'model': model_name,
            'should_retrain': len(triggers_fired) > 0,
            'triggers_fired': triggers_fired,
            'details': details,
            'checked_at': datetime.now().isoformat()
        }
    
    def should_retrain(self, model_name: str) -> bool:
        """Simple boolean check if retraining needed."""
        return self.check_conditions(model_name)['should_retrain']
    
    def trigger_retraining(self, model_name: str, reason: str = None,
                           blocking: bool = True) -> dict:
        """Launch retraining job for a model."""
        job_id = f"retrain-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        script = self.TRAINING_SCRIPTS.get(model_name)
        if not script or not Path(script).exists():
            return {'status': 'failed', 'job_id': job_id, 'error': 'Training script not found'}
        
        self._notify('retraining_started', {'model': model_name, 'reason': reason, 'job_id': job_id})
        
        start_time = time.time()
        try:
            result = subprocess.run(
                ['python', script],
                capture_output=True,
                text=True,
                timeout=3600 if blocking else 1
            )
            success = result.returncode == 0
        except subprocess.TimeoutExpired:
            if not blocking:
                return {'status': 'started', 'job_id': job_id, 'model': model_name}
            success = False
        except Exception as e:
            return {'status': 'failed', 'job_id': job_id, 'error': str(e)}
        
        duration = time.time() - start_time
        self._last_trained[model_name] = datetime.now()
        
        if success:
            # Validate new model
            validation = self.validate_new_model(model_name)
            
            if validation['passed']:
                if self.config.get('auto_deploy'):
                    self._deploy_model(model_name)
                
                self._notify('retraining_success', {
                    'model': model_name, 'job_id': job_id,
                    'duration_seconds': duration, 'validation': validation
                })
                
                return {
                    'status': 'completed', 'job_id': job_id, 'model': model_name,
                    'reason': reason, 'duration_seconds': duration,
                    'validation_passed': True
                }
            else:
                self._notify('validation_failed', {'model': model_name, 'validation': validation})
                return {'status': 'validation_failed', 'job_id': job_id, 'validation': validation}
        
        self._notify('retraining_failed', {'model': model_name, 'job_id': job_id})
        return {'status': 'failed', 'job_id': job_id, 'model': model_name}
    
    def validate_new_model(self, model_name: str) -> dict:
        """Run validation tests on newly trained model."""
        try:
            result = subprocess.run(
                ['python', 'scripts/validate_realworld.py'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Try to load validation report
            report_path = Path('evaluation/validation_report.json')
            if report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                
                pass_rate = report.get('summary', {}).get('pass_rate', 0)
                return {
                    'passed': pass_rate >= self.config.get('min_accuracy', 0.90),
                    'pass_rate': pass_rate,
                    'tests_passed': report.get('summary', {}).get('passed', 0),
                    'tests_total': report.get('summary', {}).get('total_tests', 0)
                }
        except Exception as e:
            return {'passed': False, 'error': str(e)}
        
        return {'passed': result.returncode == 0}
    
    def run_daemon(self, interval_minutes: int = 60):
        """Run as background daemon, checking conditions periodically."""
        self._stop_daemon = False
        
        def daemon_loop():
            while not self._stop_daemon:
                for model_name in self.TRAINING_SCRIPTS.keys():
                    try:
                        conditions = self.check_conditions(model_name)
                        if conditions['should_retrain']:
                            self.trigger_retraining(
                                model_name,
                                reason=','.join(conditions['triggers_fired']),
                                blocking=True
                            )
                    except Exception as e:
                        print(f"Error checking {model_name}: {e}")
                
                # Sleep in small intervals to allow stopping
                for _ in range(interval_minutes * 6):
                    if self._stop_daemon:
                        break
                    time.sleep(10)
        
        self._daemon_thread = threading.Thread(target=daemon_loop, daemon=True)
        self._daemon_thread.start()
    
    def stop_daemon(self):
        """Stop the background daemon thread."""
        self._stop_daemon = True
        if self._daemon_thread:
            self._daemon_thread.join(timeout=30)
    
    def _check_drift(self, model_name: str) -> dict:
        """Check for data drift."""
        stats = self.monitor.get_stats(model_name)
        if not stats or stats.get('count', 0) < 100:
            return {'triggered': False, 'reason': 'insufficient_data'}
        
        # Simple drift check based on confidence distribution shift
        mean_conf = stats.get('confidence_mean', 0.5)
        if abs(mean_conf - 0.5) > 0.2:  # Significant shift from expected
            return {'triggered': True, 'confidence_mean': mean_conf, 'reason': 'distribution_shift'}
        
        return {'triggered': False, 'confidence_mean': mean_conf}
    
    def _check_performance(self, model_name: str) -> dict:
        """Check if performance below threshold."""
        metrics = self.registry.get_metrics(model_name)
        if not metrics:
            return {'triggered': False, 'reason': 'no_metrics'}
        
        accuracy = metrics.get('accuracy', metrics.get('f1_score', 1.0))
        threshold = self.config.get('min_accuracy', 0.90)
        
        return {
            'triggered': accuracy < threshold,
            'current': accuracy,
            'threshold': threshold
        }
    
    def _check_schedule(self, model_name: str) -> dict:
        """Check if scheduled retraining is due."""
        last = self._last_trained.get(model_name)
        schedule_days = self.config.get('schedule_days', 7)
        
        if last is None:
            # Check model file modification time
            model_path = Path('models') / f"{model_name}.pt"
            if model_path.exists():
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                last = mtime
        
        if last is None:
            return {'triggered': False, 'reason': 'no_baseline'}
        
        days_since = (datetime.now() - last).days
        return {
            'triggered': days_since >= schedule_days,
            'days_since_training': days_since,
            'schedule_days': schedule_days,
            'last_trained': last.isoformat() if last else None
        }
    
    def _deploy_model(self, model_name: str):
        """Deploy newly trained model as active version."""
        try:
            versions = self.registry.list_versions(model_name)
            if versions:
                latest = max(versions, key=lambda v: [int(x) for x in v.split('.')])
                self.registry.set_active(model_name, latest)
        except Exception:
            pass
    
    def _notify(self, event: str, details: dict):
        """Send notification via alert_dispatcher if configured."""
        if self.alert_dispatcher:
            try:
                self.alert_dispatcher.send({
                    'severity': 'MEDIUM' if 'failed' in event else 'LOW',
                    'attack_type': f'RETRAINING_{event.upper()}',
                    'confidence': 1.0,
                    'source': details
                })
            except Exception:
                pass
