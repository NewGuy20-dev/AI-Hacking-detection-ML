"""Shared pytest fixtures for AI-Hacking-Detection-ML tests."""
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def temp_models_dir(temp_dir):
    """Create a temporary models directory."""
    models_dir = temp_dir / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def mock_sklearn_model():
    """Create a real sklearn model for testing."""
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy='constant', constant=0)
    model.fit([[0], [1]], [0, 1])  # Minimal fit
    return model


@pytest.fixture
def mock_pytorch_model():
    """Create a simple PyTorch-like model for testing."""
    import torch
    import torch.nn as nn
    
    class MockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        def forward(self, x):
            return self.fc(x.float().mean(dim=-1, keepdim=True)).squeeze(-1)
    return MockNet()


@pytest.fixture
def sample_payloads():
    """Sample payload strings for testing."""
    return [
        "SELECT * FROM users WHERE id=1",
        "<script>alert('xss')</script>",
        "normal user input text",
        "'; DROP TABLE users; --",
        "Hello, this is a benign message",
    ]


@pytest.fixture
def sample_urls():
    """Sample URL strings for testing."""
    return [
        "https://google.com",
        "http://malicious-site.com/phishing",
        "example.com/path",
        "https://legitimate-bank.com/login",
        "http://192.168.1.1/admin",
    ]


@pytest.fixture
def sample_network_features():
    """Sample network feature arrays for testing."""
    return np.random.randn(5, 41).astype(np.float32)


@pytest.fixture
def registry(temp_models_dir):
    """Create a ModelRegistry with temporary directory."""
    from src.model_registry import ModelRegistry
    return ModelRegistry(str(temp_models_dir))


# Phase B fixtures

@pytest.fixture
def mock_model_registry():
    """Create mock ModelRegistry for Phase B tests."""
    registry = Mock()
    registry.load.return_value = Mock()
    registry.list_versions.return_value = ['1.0.0', '2.0.0']
    registry.set_active.return_value = None
    registry.get_metrics.return_value = {'accuracy': 0.95, 'f1_score': 0.94}
    return registry


@pytest.fixture
def mock_model_monitor():
    """Create mock ModelMonitor for Phase B tests."""
    monitor = Mock()
    monitor.get_stats.return_value = {
        'count': 1000,
        'confidence_mean': 0.5,
        'latency_mean': 10.0,
        'latency_p95': 25.0
    }
    monitor.log_prediction.return_value = None
    return monitor


@pytest.fixture
def mock_alert_dispatcher():
    """Create mock AlertDispatcher for Phase B tests."""
    dispatcher = Mock()
    dispatcher.send.return_value = {'success': True, 'alert_id': 'ALT-TEST-001'}
    dispatcher.send_from_prediction.return_value = {'success': True}
    return dispatcher


@pytest.fixture
def mock_predictor():
    """Create mock BatchHybridPredictor for Phase B tests."""
    predictor = Mock()
    predictor.pytorch_models = {'payload_cnn': Mock(), 'url_cnn': Mock()}
    predictor.sklearn_models = {'network': Mock()}
    predictor.device = 'cpu'
    predictor.predict_batch.return_value = {
        'is_attack': np.array([True]),
        'confidence': np.array([0.95]),
        'scores': {
            'payload': np.array([0.95]),
            'url': np.array([0.5]),
            'network': np.array([0.5])
        }
    }
    return predictor


@pytest.fixture
def sample_alert():
    """Sample alert dict for testing."""
    return {
        'severity': 'HIGH',
        'attack_type': 'SQL_INJECTION',
        'confidence': 0.95,
        'id': 'ALT-TEST-001',
        'timestamp': '2025-12-21T07:00:00Z',
        'source': {'payload': "' OR 1=1--", 'ip': '192.168.1.100'}
    }


@pytest.fixture
def sample_prediction():
    """Sample prediction result for testing."""
    return {
        'is_attack': [True],
        'confidence': [0.95],
        'scores': {
            'payload': [0.95],
            'url': [0.5],
            'network': [0.5],
            'timeseries': [0.5]
        }
    }
