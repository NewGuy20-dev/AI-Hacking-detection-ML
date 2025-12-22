"""Tests for Retraining Trigger."""
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading
import time

from src.retraining_trigger import RetrainingTrigger


@pytest.fixture
def mock_registry():
    """Create mock ModelRegistry."""
    registry = Mock()
    registry.get_metrics.return_value = {'accuracy': 0.95, 'f1_score': 0.94}
    registry.list_versions.return_value = ['1.0.0', '2.0.0']
    registry.set_active.return_value = None
    return registry


@pytest.fixture
def mock_monitor():
    """Create mock ModelMonitor."""
    monitor = Mock()
    monitor.get_stats.return_value = {
        'count': 1000,
        'confidence_mean': 0.5,
        'latency_mean': 10.0
    }
    return monitor


@pytest.fixture
def mock_alert_dispatcher():
    """Create mock AlertDispatcher."""
    dispatcher = Mock()
    dispatcher.send.return_value = {'success': True}
    return dispatcher


@pytest.fixture
def trigger(mock_registry, mock_monitor, mock_alert_dispatcher):
    """Create RetrainingTrigger with mocks."""
    return RetrainingTrigger(
        registry=mock_registry,
        monitor=mock_monitor,
        alert_dispatcher=mock_alert_dispatcher
    )


class TestRetrainingTriggerConditions:
    
    def test_check_conditions_no_triggers(self, trigger):
        """Test check_conditions when no triggers fire."""
        result = trigger.check_conditions('payload_cnn')
        
        assert result['model'] == 'payload_cnn'
        assert result['should_retrain'] is False
        assert len(result['triggers_fired']) == 0
    
    def test_check_drift_insufficient_data(self, trigger, mock_monitor):
        """Test drift check with insufficient data."""
        mock_monitor.get_stats.return_value = {'count': 50}
        
        result = trigger._check_drift('payload_cnn')
        
        assert result['triggered'] is False
        assert result['reason'] == 'insufficient_data'
    
    def test_check_drift_detected(self, trigger, mock_monitor):
        """Test drift detection when distribution shifts."""
        mock_monitor.get_stats.return_value = {
            'count': 1000,
            'confidence_mean': 0.85  # Shifted from expected 0.5
        }
        
        result = trigger._check_drift('payload_cnn')
        
        assert result['triggered'] is True
        assert result['reason'] == 'distribution_shift'
    
    def test_check_performance_above_threshold(self, trigger, mock_registry):
        """Test performance check when above threshold."""
        mock_registry.get_metrics.return_value = {'accuracy': 0.95}
        
        result = trigger._check_performance('payload_cnn')
        
        assert result['triggered'] is False
    
    def test_check_performance_below_threshold(self, trigger, mock_registry):
        """Test performance check when below threshold."""
        mock_registry.get_metrics.return_value = {'accuracy': 0.85}
        trigger.config['min_accuracy'] = 0.90
        
        result = trigger._check_performance('payload_cnn')
        
        assert result['triggered'] is True
    
    def test_check_performance_no_metrics(self, trigger, mock_registry):
        """Test performance check with no metrics available."""
        mock_registry.get_metrics.return_value = {}
        
        result = trigger._check_performance('payload_cnn')
        
        assert result['triggered'] is False
        assert result['reason'] == 'no_metrics'
    
    def test_check_schedule_not_due(self, trigger, tmp_path):
        """Test schedule check when not due."""
        # Create recent model file
        model_path = tmp_path / "models" / "payload_cnn.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.touch()
        
        trigger.config['schedule_days'] = 7
        trigger._last_trained['payload_cnn'] = datetime.now()
        
        result = trigger._check_schedule('payload_cnn')
        
        assert result['triggered'] is False
    
    def test_check_schedule_due(self, trigger):
        """Test schedule check when due."""
        trigger.config['schedule_days'] = 7
        trigger._last_trained['payload_cnn'] = datetime.now() - timedelta(days=10)
        
        result = trigger._check_schedule('payload_cnn')
        
        assert result['triggered'] is True
        assert result['days_since_training'] >= 7
    
    def test_should_retrain_simple(self, trigger, mock_monitor):
        """Test simple should_retrain check."""
        mock_monitor.get_stats.return_value = {'count': 1000, 'confidence_mean': 0.5}
        
        result = trigger.should_retrain('payload_cnn')
        
        assert isinstance(result, bool)


class TestRetrainingTriggerExecution:
    
    @patch('src.retraining_trigger.subprocess.run')
    def test_trigger_retraining_script_not_found(self, mock_run, trigger):
        """Test retraining with missing script."""
        result = trigger.trigger_retraining('nonexistent_model')
        
        assert result['status'] == 'failed'
        assert 'not found' in result['error']
    
    @patch('src.retraining_trigger.subprocess.run')
    @patch('src.retraining_trigger.Path.exists')
    def test_trigger_retraining_success(self, mock_exists, mock_run, trigger):
        """Test successful retraining."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)
        
        # Mock validation
        trigger.validate_new_model = Mock(return_value={'passed': True, 'pass_rate': 0.95})
        
        result = trigger.trigger_retraining('payload_cnn', reason='test')
        
        assert result['status'] == 'completed'
        assert result['validation_passed'] is True
    
    @patch('src.retraining_trigger.subprocess.run')
    @patch('src.retraining_trigger.Path.exists')
    def test_trigger_retraining_validation_failed(self, mock_exists, mock_run, trigger):
        """Test retraining with failed validation."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=0)
        
        trigger.validate_new_model = Mock(return_value={'passed': False, 'pass_rate': 0.80})
        
        result = trigger.trigger_retraining('payload_cnn')
        
        assert result['status'] == 'validation_failed'
    
    @patch('src.retraining_trigger.subprocess.run')
    @patch('src.retraining_trigger.Path.exists')
    def test_trigger_retraining_script_failed(self, mock_exists, mock_run, trigger):
        """Test retraining when script fails."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(returncode=1)
        
        result = trigger.trigger_retraining('payload_cnn')
        
        assert result['status'] == 'failed'


class TestRetrainingTriggerValidation:
    
    @patch('src.retraining_trigger.subprocess.run')
    def test_validate_new_model_success(self, mock_run, trigger, tmp_path):
        """Test successful model validation."""
        mock_run.return_value = Mock(returncode=0)
        
        # Create mock validation report
        eval_dir = Path('evaluation')
        eval_dir.mkdir(exist_ok=True)
        report = {
            'summary': {
                'pass_rate': 0.95,
                'passed': 40,
                'total_tests': 42
            }
        }
        with open(eval_dir / 'validation_report.json', 'w') as f:
            json.dump(report, f)
        
        result = trigger.validate_new_model('payload_cnn')
        
        assert result['passed'] is True
        assert result['pass_rate'] == 0.95
    
    @patch('src.retraining_trigger.subprocess.run')
    def test_validate_new_model_below_threshold(self, mock_run, trigger):
        """Test validation fails when below threshold."""
        mock_run.return_value = Mock(returncode=0)
        
        # Create mock validation report with low pass rate
        eval_dir = Path('evaluation')
        eval_dir.mkdir(exist_ok=True)
        report = {'summary': {'pass_rate': 0.80, 'passed': 34, 'total_tests': 42}}
        with open(eval_dir / 'validation_report.json', 'w') as f:
            json.dump(report, f)
        
        trigger.config['min_accuracy'] = 0.90
        result = trigger.validate_new_model('payload_cnn')
        
        assert result['passed'] is False
    
    @patch('src.retraining_trigger.subprocess.run')
    def test_validate_new_model_exception(self, mock_run, trigger):
        """Test validation handles exceptions."""
        mock_run.side_effect = Exception("Validation error")
        
        result = trigger.validate_new_model('payload_cnn')
        
        assert result['passed'] is False
        assert 'error' in result


class TestRetrainingTriggerDaemon:
    
    def test_run_daemon_starts_thread(self, trigger):
        """Test daemon starts background thread."""
        trigger.run_daemon(interval_minutes=1)
        
        assert trigger._daemon_thread is not None
        assert trigger._daemon_thread.is_alive()
        
        trigger.stop_daemon()
    
    def test_stop_daemon(self, trigger):
        """Test stopping daemon thread."""
        trigger.run_daemon(interval_minutes=1)
        trigger.stop_daemon()
        
        assert trigger._stop_daemon is True
        # Thread should stop within timeout
        trigger._daemon_thread.join(timeout=5)


class TestRetrainingTriggerNotifications:
    
    def test_notify_sends_alert(self, trigger, mock_alert_dispatcher):
        """Test notifications are sent via alert dispatcher."""
        trigger._notify('retraining_started', {'model': 'test', 'reason': 'drift'})
        
        mock_alert_dispatcher.send.assert_called_once()
    
    def test_notify_handles_exception(self, trigger, mock_alert_dispatcher):
        """Test notification handles dispatcher exceptions."""
        mock_alert_dispatcher.send.side_effect = Exception("Send failed")
        
        # Should not raise
        trigger._notify('test_event', {'data': 'test'})
    
    def test_notify_without_dispatcher(self, mock_registry, mock_monitor):
        """Test notification works without dispatcher."""
        trigger = RetrainingTrigger(
            registry=mock_registry,
            monitor=mock_monitor,
            alert_dispatcher=None
        )
        
        # Should not raise
        trigger._notify('test_event', {'data': 'test'})


class TestRetrainingTriggerEdgeCases:
    
    def test_check_conditions_unknown_model(self, trigger):
        """Test check_conditions for unknown model."""
        result = trigger.check_conditions('unknown_model')
        
        assert result['model'] == 'unknown_model'
        # Should not crash, just return no triggers
    
    def test_default_config(self, mock_registry, mock_monitor):
        """Test default configuration values."""
        trigger = RetrainingTrigger(registry=mock_registry, monitor=mock_monitor)
        
        assert trigger.config['drift_threshold'] == 0.05
        assert trigger.config['min_accuracy'] == 0.90
        assert trigger.config['schedule_days'] == 7
    
    def test_custom_config(self, mock_registry, mock_monitor):
        """Test custom configuration."""
        custom_config = {
            'drift_threshold': 0.10,
            'min_accuracy': 0.95,
            'schedule_days': 14
        }
        trigger = RetrainingTrigger(
            registry=mock_registry,
            monitor=mock_monitor,
            config=custom_config
        )
        
        assert trigger.config['drift_threshold'] == 0.10
        assert trigger.config['schedule_days'] == 14
    
    def test_training_scripts_mapping(self, trigger):
        """Test training scripts are mapped correctly."""
        assert 'payload_cnn' in trigger.TRAINING_SCRIPTS
        assert 'url_cnn' in trigger.TRAINING_SCRIPTS
        assert 'timeseries_lstm' in trigger.TRAINING_SCRIPTS
    
    def test_concurrent_condition_checks(self, trigger, mock_monitor):
        """Test concurrent condition checks don't interfere."""
        mock_monitor.get_stats.return_value = {'count': 1000, 'confidence_mean': 0.5}
        
        results = []
        
        def check():
            result = trigger.check_conditions('payload_cnn')
            results.append(result)
        
        threads = [threading.Thread(target=check) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(results) == 10
        assert all(r['model'] == 'payload_cnn' for r in results)
    
    def test_deploy_model_no_versions(self, trigger, mock_registry):
        """Test deploy handles no versions gracefully."""
        mock_registry.list_versions.return_value = []
        
        # Should not raise
        trigger._deploy_model('payload_cnn')
    
    def test_check_schedule_no_baseline(self, trigger):
        """Test schedule check with no baseline."""
        # No last_trained and no model file
        result = trigger._check_schedule('nonexistent_model')
        
        assert result['triggered'] is False
        assert result['reason'] == 'no_baseline'
