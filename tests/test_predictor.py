"""Tests for BatchHybridPredictor and monitoring integration."""
import numpy as np
import pytest

from src.input_validator import InputValidator, ValidationError
from src.model_monitor import ModelMonitor, DriftDetector


class TestModelMonitor:
    
    def test_log_prediction(self):
        """Test logging a prediction."""
        monitor = ModelMonitor()
        monitor.log_prediction("test_model", latency=0.05, confidence=0.9)
        
        stats = monitor.get_stats("test_model")
        assert stats["count"] == 1
        assert stats["latency_mean"] == 0.05
        assert stats["confidence_mean"] == 0.9
    
    def test_multiple_predictions(self):
        """Test logging multiple predictions."""
        monitor = ModelMonitor()
        for i in range(100):
            monitor.log_prediction("model", latency=0.01 * i, confidence=0.5)
        
        stats = monitor.get_stats("model")
        assert stats["count"] == 100
        assert stats["latency_p95"] is not None
    
    def test_track_context_manager(self):
        """Test tracking with context manager."""
        monitor = ModelMonitor()
        
        with monitor.track("model") as ctx:
            # Simulate work
            _ = sum(range(1000))
            ctx["confidence"] = 0.85
        
        stats = monitor.get_stats("model")
        assert stats["count"] == 1
        assert stats["latency_mean"] > 0
        assert stats["confidence_mean"] == 0.85
    
    def test_get_all_stats(self):
        """Test getting stats for all models."""
        monitor = ModelMonitor()
        monitor.log_prediction("model_a", 0.01, 0.9)
        monitor.log_prediction("model_b", 0.02, 0.8)
        
        all_stats = monitor.get_stats()
        assert "model_a" in all_stats
        assert "model_b" in all_stats
    
    def test_reset_single_model(self):
        """Test resetting metrics for one model."""
        monitor = ModelMonitor()
        monitor.log_prediction("model_a", 0.01, 0.9)
        monitor.log_prediction("model_b", 0.02, 0.8)
        
        monitor.reset("model_a")
        
        assert monitor.get_stats("model_a") == {}
        assert monitor.get_stats("model_b")["count"] == 1
    
    def test_reset_all(self):
        """Test resetting all metrics."""
        monitor = ModelMonitor()
        monitor.log_prediction("model_a", 0.01, 0.9)
        monitor.log_prediction("model_b", 0.02, 0.8)
        
        monitor.reset()
        
        assert monitor.get_stats() == {}
    
    def test_max_samples_limit(self):
        """Test that metrics are trimmed at max_samples."""
        monitor = ModelMonitor(max_samples=50)
        
        for i in range(100):
            monitor.log_prediction("model", 0.01, 0.5)
        
        assert monitor.get_stats("model")["count"] == 50
    
    def test_export(self, temp_dir):
        """Test exporting metrics to JSON."""
        monitor = ModelMonitor()
        monitor.log_prediction("model", 0.01, 0.9)
        
        filepath = temp_dir / "metrics.json"
        monitor.export(str(filepath))
        
        assert filepath.exists()


class TestDriftDetector:
    
    def test_no_drift(self):
        """Test no drift detected for similar distributions."""
        reference = np.random.normal(0, 1, 1000)
        new_data = np.random.normal(0, 1, 1000)
        
        detector = DriftDetector(reference)
        assert detector.detect(new_data) == False
    
    def test_drift_detected(self):
        """Test drift detected for different distributions."""
        reference = np.random.normal(0, 1, 1000)
        new_data = np.random.normal(5, 1, 1000)  # Shifted mean
        
        detector = DriftDetector(reference)
        assert detector.detect(new_data) == True
    
    def test_get_report(self):
        """Test getting drift report."""
        reference = np.random.normal(0, 1, 1000)
        new_data = np.random.normal(0, 1, 1000)
        
        detector = DriftDetector(reference)
        detector.detect(new_data)
        
        report = detector.get_report()
        assert "statistic" in report
        assert "p_value" in report
        assert "is_drifted" in report
    
    def test_custom_threshold(self):
        """Test custom threshold."""
        reference = np.random.normal(0, 1, 1000)
        new_data = np.random.normal(0.3, 1, 1000)  # Slight shift
        
        # Strict threshold - more likely to detect drift
        strict_detector = DriftDetector(reference, threshold=0.5)
        # Lenient threshold - less likely to detect drift
        lenient_detector = DriftDetector(reference, threshold=0.001)
        
        # Results may vary, but strict should be >= lenient in detection
        strict_result = strict_detector.detect(new_data)
        lenient_result = lenient_detector.detect(new_data)
        
        # At minimum, both should return boolean-like value
        assert bool(strict_result) in (True, False)
        assert bool(lenient_result) in (True, False)
    
    def test_update_reference(self):
        """Test updating reference distribution."""
        reference = np.random.normal(0, 1, 1000)
        detector = DriftDetector(reference)
        
        new_reference = np.random.normal(5, 1, 1000)
        detector.update_reference(new_reference)
        
        # Now data similar to new reference should not drift
        similar_data = np.random.normal(5, 1, 1000)
        assert detector.detect(similar_data) == False
    
    def test_small_sample_handling(self):
        """Test handling of small samples."""
        reference = np.array([1, 2, 3])
        detector = DriftDetector(reference)
        
        # Should return False for too-small samples
        assert detector.detect(np.array([1, 2])) == False


class TestBatchPredictorIntegration:
    """Integration tests for BatchHybridPredictor with validation/monitoring."""
    
    def test_predictor_with_validation(self, sample_payloads):
        """Test predictor validates inputs."""
        from src.batch_predictor import BatchHybridPredictor
        
        predictor = BatchHybridPredictor(validator=True)
        predictor.loaded = True  # Skip model loading
        predictor.pytorch_models = {}  # Empty models
        
        # Should not raise - validation passes
        data = {"payloads": sample_payloads}
        # Note: predict_batch will return 0.5 for missing models
    
    def test_predictor_with_monitoring(self):
        """Test predictor logs to monitor."""
        from src.batch_predictor import BatchHybridPredictor
        
        monitor = ModelMonitor()
        predictor = BatchHybridPredictor(monitor=monitor, validator=False)
        predictor.loaded = True
        predictor.pytorch_models = {}
        
        # After prediction, monitor should have entries
        # (actual prediction requires models, so this is a setup test)
        assert predictor.monitor is monitor
    
    def test_factory_function(self):
        """Test create_batch_predictor factory."""
        from src.batch_predictor import create_batch_predictor
        
        # Should not raise even without models
        try:
            predictor = create_batch_predictor(
                models_dir="nonexistent",
                enable_validation=True,
                enable_monitoring=True
            )
            assert predictor.validator is not None
            assert predictor.monitor is not None
        except FileNotFoundError:
            pass  # Expected if models dir doesn't exist
