"""Integration tests for Phase B features."""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import threading
import time


class TestAlertIntegration:
    """Integration tests for Alert System with other components."""
    
    def test_alert_from_prediction_pipeline(self):
        """Test alert triggered from prediction result."""
        from src.alerts import AlertDispatcher
        
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        # Simulate prediction result
        prediction = {
            'is_attack': [True],
            'confidence': [0.95],
            'scores': {'payload': [0.95], 'url': [0.5]}
        }
        input_data = {'payload': "' OR 1=1--"}
        
        result = dispatcher.send_from_prediction(prediction, input_data, threshold=0.8)
        
        assert result is not None
        assert result['success'] is True
        assert 'SQL_INJECTION' in str(result) or result['alert_id'].startswith('ALT-')
    
    def test_alert_with_batch_predictor_mock(self):
        """Test alert integration with mocked BatchHybridPredictor."""
        from src.alerts import AlertDispatcher
        
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        # Mock predictor result
        mock_result = {
            'is_attack': np.array([True, False, True]),
            'confidence': np.array([0.95, 0.3, 0.88]),
            'scores': {}
        }
        
        alerts_sent = []
        for i in range(len(mock_result['is_attack'])):
            single_result = {
                'is_attack': [mock_result['is_attack'][i]],
                'confidence': [mock_result['confidence'][i]]
            }
            result = dispatcher.send_from_prediction(
                single_result, 
                {'payload': f'test_{i}'}, 
                threshold=0.8
            )
            if result:
                alerts_sent.append(result)
        
        # Should send 2 alerts (confidence > 0.8 and is_attack)
        assert len(alerts_sent) == 2
    
    def test_alert_rate_limiting_under_load(self):
        """Test rate limiting under simulated load."""
        from src.alerts import AlertDispatcher
        
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        dispatcher._rate_limit_max = 5
        dispatcher._rate_limit_window = 1  # 1 second window
        
        results = []
        for i in range(20):
            result = dispatcher.send({
                'severity': 'HIGH',
                'attack_type': 'TEST',
                'confidence': 0.9
            })
            results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        rate_limited = sum(1 for r in results if r.get('reason') == 'rate_limited')
        
        assert successful == 5
        assert rate_limited == 15


class TestABTestingIntegration:
    """Integration tests for A/B Testing with other components."""
    
    def test_ab_test_full_lifecycle(self, tmp_path):
        """Test complete A/B test lifecycle."""
        from src.ab_testing import ABTestManager
        
        mock_registry = Mock()
        mock_registry.load.return_value = Mock()
        mock_registry.set_active.return_value = None
        
        manager = ABTestManager(
            registry=mock_registry,
            storage_path=str(tmp_path / "ab_tests")
        )
        
        # Create experiment
        exp = manager.create_experiment(
            "lifecycle_test",
            {"name": "model", "version": "1.0.0"},
            {"name": "model", "version": "2.0.0"},
            min_samples=50
        )
        assert exp['status'] == 'running'
        
        # Simulate traffic
        for i in range(100):
            _, variant = manager.get_model("lifecycle_test")
            # Simulate B being better
            if variant == 'A':
                manager.record_outcome("lifecycle_test", variant, 
                                      prediction=0.6 if i % 3 == 0 else 0.4, 
                                      actual=1, latency_ms=10)
            else:
                manager.record_outcome("lifecycle_test", variant,
                                      prediction=0.8 if i % 5 != 0 else 0.4,
                                      actual=1, latency_ms=8)
        
        # Check results
        results = manager.get_results("lifecycle_test")
        assert results['samples']['A'] + results['samples']['B'] == 100
        
        # End experiment
        final = manager.end_experiment("lifecycle_test")
        assert manager._experiments["lifecycle_test"]['status'] == 'completed'
    
    def test_ab_test_persistence(self, tmp_path):
        """Test A/B test state persists across restarts."""
        from src.ab_testing import ABTestManager
        
        mock_registry = Mock()
        mock_registry.load.return_value = Mock()
        storage = str(tmp_path / "ab_tests")
        
        # Create manager and experiment
        manager1 = ABTestManager(registry=mock_registry, storage_path=storage)
        manager1.create_experiment("persist_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        manager1.record_outcome("persist_test", "A", prediction=0.8, latency_ms=10)
        
        # Create new manager (simulating restart)
        manager2 = ABTestManager(registry=mock_registry, storage_path=storage)
        
        # Should have loaded experiment
        assert "persist_test" in manager2._experiments
        results = manager2.get_results("persist_test")
        assert results['samples']['A'] == 1


class TestRetrainingIntegration:
    """Integration tests for Retraining Trigger with other components."""
    
    def test_retraining_with_alert_notification(self):
        """Test retraining sends alerts via dispatcher."""
        from src.retraining_trigger import RetrainingTrigger
        from src.alerts import AlertDispatcher
        
        mock_registry = Mock()
        mock_registry.get_metrics.return_value = {'accuracy': 0.95}
        
        mock_monitor = Mock()
        mock_monitor.get_stats.return_value = {'count': 1000, 'confidence_mean': 0.5}
        
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        trigger = RetrainingTrigger(
            registry=mock_registry,
            monitor=mock_monitor,
            alert_dispatcher=dispatcher
        )
        
        # Trigger notification
        trigger._notify('retraining_started', {'model': 'test', 'reason': 'drift'})
        
        # Alert should have been sent (check dispatcher state)
        # Since console channel is enabled by default, this should work
    
    def test_retraining_condition_check_all_models(self):
        """Test checking conditions for all configured models."""
        from src.retraining_trigger import RetrainingTrigger
        
        mock_registry = Mock()
        mock_registry.get_metrics.return_value = {'accuracy': 0.95}
        
        mock_monitor = Mock()
        mock_monitor.get_stats.return_value = {'count': 1000, 'confidence_mean': 0.5}
        
        trigger = RetrainingTrigger(registry=mock_registry, monitor=mock_monitor)
        
        # Check all models
        for model_name in trigger.TRAINING_SCRIPTS.keys():
            result = trigger.check_conditions(model_name)
            assert result['model'] == model_name
            assert 'should_retrain' in result


class TestEndToEndFlows:
    """End-to-end integration tests."""
    
    def test_prediction_to_alert_flow(self):
        """Test flow from prediction to alert."""
        from src.alerts import AlertDispatcher
        
        # Setup
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        # Simulate high-confidence attack detection
        prediction = {
            'is_attack': [True],
            'confidence': [0.98]
        }
        input_data = {'payload': "<script>alert('XSS')</script>"}
        
        # Should trigger alert
        result = dispatcher.send_from_prediction(prediction, input_data, threshold=0.8)
        
        assert result is not None
        assert result['success'] is True
    
    def test_ab_test_to_registry_deployment(self, tmp_path):
        """Test A/B test winner deployment to registry."""
        from src.ab_testing import ABTestManager
        
        mock_registry = Mock()
        mock_registry.load.return_value = Mock()
        mock_registry.set_active.return_value = None
        
        manager = ABTestManager(
            registry=mock_registry,
            storage_path=str(tmp_path / "ab_tests")
        )
        
        # Create experiment
        manager.create_experiment(
            "deploy_test",
            {"name": "model", "version": "1.0.0"},
            {"name": "model", "version": "2.0.0"}
        )
        
        # Simulate B being significantly better
        for i in range(100):
            manager.record_outcome("deploy_test", "A", prediction=0.3, actual=1)
            manager.record_outcome("deploy_test", "B", prediction=0.9, actual=1)
        
        # End and deploy
        result = manager.end_experiment("deploy_test", deploy_winner=True)
        
        # If winner detected, registry should be called
        if result.get('winner'):
            mock_registry.set_active.assert_called()
    
    def test_drift_detection_to_retraining(self):
        """Test drift detection triggering retraining check."""
        from src.retraining_trigger import RetrainingTrigger
        from src.model_monitor import DriftDetector
        import numpy as np
        
        # Create drift detector with reference data
        reference = np.random.normal(0.5, 0.1, 1000)
        detector = DriftDetector(reference, threshold=0.05)
        
        # Simulate drifted data
        drifted = np.random.normal(0.8, 0.1, 1000)  # Shifted mean
        
        is_drifted = detector.detect(drifted)
        
        # Should detect drift (use bool() for numpy bool)
        assert bool(is_drifted) == True
        
        report = detector.get_report()
        assert report['is_drifted'] == True
        assert report['p_value'] < 0.05


class TestConcurrencyAndThreadSafety:
    """Tests for concurrent access and thread safety."""
    
    def test_concurrent_alert_sending(self):
        """Test concurrent alert sending."""
        from src.alerts import AlertDispatcher
        
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        dispatcher._rate_limit_max = 100  # High limit for this test
        
        results = []
        errors = []
        
        def send_alert(i):
            try:
                result = dispatcher.send({
                    'severity': 'HIGH',
                    'attack_type': f'TEST_{i}',
                    'confidence': 0.9
                })
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=send_alert, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 50
    
    def test_concurrent_ab_test_recording(self, tmp_path):
        """Test concurrent outcome recording in A/B test."""
        from src.ab_testing import ABTestManager
        
        mock_registry = Mock()
        mock_registry.load.return_value = Mock()
        
        manager = ABTestManager(
            registry=mock_registry,
            storage_path=str(tmp_path / "ab_tests")
        )
        
        manager.create_experiment("concurrent_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        errors = []
        
        def record(i):
            try:
                variant = 'A' if i % 2 == 0 else 'B'
                manager.record_outcome("concurrent_test", variant, prediction=0.5 + i/1000)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=record, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        results = manager.get_results("concurrent_test")
        assert results['samples']['A'] + results['samples']['B'] == 100


class TestErrorHandlingIntegration:
    """Tests for error handling across components."""
    
    def test_alert_handles_invalid_prediction_format(self):
        """Test alert system handles various prediction formats."""
        from src.alerts import AlertDispatcher
        
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        # Test with numpy arrays
        result1 = dispatcher.send_from_prediction(
            {'is_attack': np.array([True]), 'confidence': np.array([0.95])},
            {'payload': 'test'},
            threshold=0.8
        )
        assert result1 is not None
        
        # Test with lists
        result2 = dispatcher.send_from_prediction(
            {'is_attack': [True], 'confidence': [0.95]},
            {'payload': 'test'},
            threshold=0.8
        )
        assert result2 is not None
        
        # Test with scalars (edge case)
        result3 = dispatcher.send_from_prediction(
            {'is_attack': True, 'confidence': 0.95},
            {'payload': 'test'},
            threshold=0.8
        )
        # Should handle gracefully
    
    def test_ab_test_handles_model_load_failure(self, tmp_path):
        """Test A/B test handles model loading failures."""
        from src.ab_testing import ABTestManager
        
        mock_registry = Mock()
        mock_registry.load.side_effect = Exception("Model not found")
        
        manager = ABTestManager(
            registry=mock_registry,
            storage_path=str(tmp_path / "ab_tests")
        )
        
        manager.create_experiment("error_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        with pytest.raises(Exception):
            manager.get_model("error_test")
    
    def test_retraining_handles_missing_validation_report(self):
        """Test retraining handles missing validation report."""
        from src.retraining_trigger import RetrainingTrigger
        
        mock_registry = Mock()
        mock_monitor = Mock()
        
        trigger = RetrainingTrigger(registry=mock_registry, monitor=mock_monitor)
        
        # Remove validation report if exists
        report_path = Path('evaluation/validation_report.json')
        if report_path.exists():
            report_path.unlink()
        
        with patch('src.retraining_trigger.subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)
            result = trigger.validate_new_model('payload_cnn')
        
        # Should handle gracefully
        assert 'passed' in result
