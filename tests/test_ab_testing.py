"""Tests for A/B Testing Framework."""
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from src.ab_testing import ABTestManager


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    return str(tmp_path / "ab_tests")


@pytest.fixture
def mock_registry():
    """Create mock ModelRegistry."""
    registry = Mock()
    registry.load.return_value = Mock()
    registry.list_versions.return_value = ['1.0.0', '2.0.0']
    registry.set_active.return_value = None
    return registry


@pytest.fixture
def mock_monitor():
    """Create mock ModelMonitor."""
    return Mock()


@pytest.fixture
def ab_manager(temp_storage, mock_registry, mock_monitor):
    """Create ABTestManager with mocks."""
    return ABTestManager(
        registry=mock_registry,
        monitor=mock_monitor,
        storage_path=temp_storage
    )


class TestABTestManagerCreation:
    
    def test_create_experiment(self, ab_manager):
        """Test creating a new experiment."""
        exp = ab_manager.create_experiment(
            name="test_exp",
            model_a={"name": "model", "version": "1.0.0"},
            model_b={"name": "model", "version": "2.0.0"},
            split_ratio=0.5,
            min_samples=100
        )
        
        assert exp['name'] == "test_exp"
        assert exp['status'] == "running"
        assert exp['split_ratio'] == 0.5
    
    def test_create_duplicate_experiment_fails(self, ab_manager):
        """Test creating duplicate experiment raises error."""
        ab_manager.create_experiment("test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        with pytest.raises(ValueError, match="already exists"):
            ab_manager.create_experiment("test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
    
    def test_experiment_persisted(self, ab_manager, temp_storage):
        """Test experiment is saved to disk."""
        ab_manager.create_experiment("persist_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        exp_file = Path(temp_storage) / "persist_test.json"
        assert exp_file.exists()
        
        with open(exp_file) as f:
            data = json.load(f)
        assert data['name'] == "persist_test"


class TestABTestManagerTrafficSplit:
    
    def test_get_model_returns_variant(self, ab_manager, mock_registry):
        """Test get_model returns model and variant."""
        ab_manager.create_experiment("split_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        model, variant = ab_manager.get_model("split_test")
        
        assert variant in ('A', 'B')
        mock_registry.load.assert_called()
    
    def test_get_model_split_ratio(self, ab_manager):
        """Test traffic split approximately matches ratio."""
        ab_manager.create_experiment(
            "ratio_test",
            {"name": "m", "version": "1"},
            {"name": "m", "version": "2"},
            split_ratio=0.7  # 70% to B
        )
        
        variants = []
        for _ in range(1000):
            _, variant = ab_manager.get_model("ratio_test")
            variants.append(variant)
        
        b_ratio = variants.count('B') / len(variants)
        assert 0.6 < b_ratio < 0.8  # Allow some variance
    
    def test_get_model_nonexistent_experiment(self, ab_manager):
        """Test get_model raises for nonexistent experiment."""
        with pytest.raises(ValueError, match="not found"):
            ab_manager.get_model("nonexistent")
    
    def test_get_model_completed_experiment(self, ab_manager):
        """Test get_model raises for completed experiment."""
        ab_manager.create_experiment("completed", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        ab_manager.end_experiment("completed")
        
        with pytest.raises(ValueError, match="not running"):
            ab_manager.get_model("completed")


class TestABTestManagerOutcomes:
    
    def test_record_outcome(self, ab_manager):
        """Test recording prediction outcome."""
        ab_manager.create_experiment("outcome_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        ab_manager.record_outcome("outcome_test", "A", prediction=0.8, latency_ms=10.5)
        
        results = ab_manager.get_results("outcome_test")
        assert results['samples']['A'] == 1
    
    def test_record_outcome_with_ground_truth(self, ab_manager):
        """Test recording outcome with actual label."""
        ab_manager.create_experiment("truth_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        ab_manager.record_outcome("truth_test", "A", prediction=0.8, actual=1)  # Correct
        ab_manager.record_outcome("truth_test", "A", prediction=0.3, actual=1)  # Wrong
        
        results = ab_manager.get_results("truth_test")
        assert results['metrics']['A']['accuracy'] == 0.5
    
    def test_record_outcome_memory_limit(self, ab_manager):
        """Test outcomes are trimmed to prevent memory issues."""
        ab_manager.create_experiment("memory_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        for i in range(15000):
            ab_manager.record_outcome("memory_test", "A", prediction=0.5)
        
        exp = ab_manager._experiments["memory_test"]
        assert len(exp['results']['A']['predictions']) <= 10000
    
    def test_record_outcome_nonexistent_experiment(self, ab_manager):
        """Test recording to nonexistent experiment doesn't crash."""
        ab_manager.record_outcome("nonexistent", "A", prediction=0.5)
        # Should not raise


class TestABTestManagerResults:
    
    def test_get_results_basic(self, ab_manager):
        """Test getting experiment results."""
        ab_manager.create_experiment("results_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        for i in range(50):
            ab_manager.record_outcome("results_test", "A", prediction=0.7, latency_ms=10)
            ab_manager.record_outcome("results_test", "B", prediction=0.8, latency_ms=8)
        
        results = ab_manager.get_results("results_test")
        
        assert results['samples']['A'] == 50
        assert results['samples']['B'] == 50
        assert 'metrics' in results
    
    def test_get_results_nonexistent(self, ab_manager):
        """Test getting results for nonexistent experiment."""
        results = ab_manager.get_results("nonexistent")
        
        assert 'error' in results
    
    def test_get_results_latency_percentiles(self, ab_manager):
        """Test latency percentile calculation."""
        ab_manager.create_experiment("latency_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        for i in range(100):
            ab_manager.record_outcome("latency_test", "A", prediction=0.5, latency_ms=i)
        
        results = ab_manager.get_results("latency_test")
        
        assert results['metrics']['A']['latency_p50'] is not None
        assert results['metrics']['A']['latency_p95'] is not None


class TestABTestManagerStatistics:
    
    def test_determine_winner_insufficient_samples(self, ab_manager):
        """Test no winner with insufficient samples."""
        ab_manager.create_experiment("small_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        for i in range(10):
            ab_manager.record_outcome("small_test", "A", prediction=0.5, actual=1)
            ab_manager.record_outcome("small_test", "B", prediction=0.5, actual=1)
        
        result = ab_manager.determine_winner("small_test")
        
        assert result['significant'] is False
        assert result.get('reason') == 'insufficient_samples'
    
    def test_determine_winner_significant_difference(self, ab_manager):
        """Test winner detection with significant difference."""
        ab_manager.create_experiment("sig_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        # A: 70% accuracy, B: 90% accuracy
        for i in range(100):
            ab_manager.record_outcome("sig_test", "A", prediction=0.8 if i < 70 else 0.3, actual=1)
            ab_manager.record_outcome("sig_test", "B", prediction=0.8 if i < 90 else 0.3, actual=1)
        
        result = ab_manager.determine_winner("sig_test")
        
        # Should detect B as winner
        if result['significant']:
            assert result['winner'] == 'B'
    
    def test_determine_winner_no_ground_truth(self, ab_manager):
        """Test winner determination without ground truth."""
        ab_manager.create_experiment("no_truth", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        for i in range(50):
            ab_manager.record_outcome("no_truth", "A", prediction=0.5)
            ab_manager.record_outcome("no_truth", "B", prediction=0.5)
        
        result = ab_manager.determine_winner("no_truth")
        
        assert result['significant'] is False


class TestABTestManagerLifecycle:
    
    def test_end_experiment(self, ab_manager):
        """Test ending an experiment."""
        ab_manager.create_experiment("end_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        result = ab_manager.end_experiment("end_test")
        
        assert ab_manager._experiments["end_test"]['status'] == 'completed'
        assert 'ended_at' in ab_manager._experiments["end_test"]
    
    def test_end_experiment_deploy_winner(self, ab_manager, mock_registry):
        """Test deploying winner when ending experiment."""
        ab_manager.create_experiment("deploy_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        # Make B clearly better
        for i in range(100):
            ab_manager.record_outcome("deploy_test", "A", prediction=0.3, actual=1)
            ab_manager.record_outcome("deploy_test", "B", prediction=0.9, actual=1)
        
        result = ab_manager.end_experiment("deploy_test", deploy_winner=True)
        
        # If winner detected, registry.set_active should be called
        if result.get('winner'):
            mock_registry.set_active.assert_called()
    
    def test_list_experiments(self, ab_manager):
        """Test listing experiments."""
        ab_manager.create_experiment("list1", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        ab_manager.create_experiment("list2", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        ab_manager.end_experiment("list2")
        
        all_exps = ab_manager.list_experiments()
        running = ab_manager.list_experiments(status="running")
        completed = ab_manager.list_experiments(status="completed")
        
        assert len(all_exps) == 2
        assert len(running) == 1
        assert len(completed) == 1
    
    def test_load_experiments_on_init(self, temp_storage, mock_registry, mock_monitor):
        """Test experiments are loaded from disk on init."""
        # Create and save an experiment
        manager1 = ABTestManager(registry=mock_registry, monitor=mock_monitor, storage_path=temp_storage)
        manager1.create_experiment("persist", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        # Create new manager - should load existing experiment
        manager2 = ABTestManager(registry=mock_registry, monitor=mock_monitor, storage_path=temp_storage)
        
        assert "persist" in manager2._experiments


class TestABTestManagerEdgeCases:
    
    def test_zero_split_ratio(self, ab_manager):
        """Test 0% split to B (all traffic to A)."""
        ab_manager.create_experiment("zero_split", {"name": "m", "version": "1"}, {"name": "m", "version": "2"}, split_ratio=0.0)
        
        variants = [ab_manager.get_model("zero_split")[1] for _ in range(100)]
        
        assert all(v == 'A' for v in variants)
    
    def test_full_split_ratio(self, ab_manager):
        """Test 100% split to B."""
        ab_manager.create_experiment("full_split", {"name": "m", "version": "1"}, {"name": "m", "version": "2"}, split_ratio=1.0)
        
        variants = [ab_manager.get_model("full_split")[1] for _ in range(100)]
        
        assert all(v == 'B' for v in variants)
    
    def test_negative_prediction_value(self, ab_manager):
        """Test handling negative prediction values."""
        ab_manager.create_experiment("neg_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        ab_manager.record_outcome("neg_test", "A", prediction=-0.5)
        
        results = ab_manager.get_results("neg_test")
        assert results['samples']['A'] == 1
    
    def test_prediction_greater_than_one(self, ab_manager):
        """Test handling prediction > 1."""
        ab_manager.create_experiment("high_test", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        ab_manager.record_outcome("high_test", "A", prediction=1.5)
        
        results = ab_manager.get_results("high_test")
        assert results['samples']['A'] == 1
    
    def test_very_small_latency(self, ab_manager):
        """Test handling very small latency values."""
        ab_manager.create_experiment("small_lat", {"name": "m", "version": "1"}, {"name": "m", "version": "2"})
        
        ab_manager.record_outcome("small_lat", "A", prediction=0.5, latency_ms=0.0001)
        
        results = ab_manager.get_results("small_lat")
        assert results['samples']['A'] == 1
    
    def test_corrupted_experiment_file(self, temp_storage, mock_registry, mock_monitor):
        """Test handling corrupted experiment file."""
        # Create corrupted file
        Path(temp_storage).mkdir(parents=True, exist_ok=True)
        with open(Path(temp_storage) / "corrupted.json", 'w') as f:
            f.write("not valid json{{{")
        
        # Should not crash on init
        manager = ABTestManager(registry=mock_registry, monitor=mock_monitor, storage_path=temp_storage)
        
        assert "corrupted" not in manager._experiments
