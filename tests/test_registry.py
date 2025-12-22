"""Tests for ModelRegistry and ModelVersion."""
import json
import pytest
from pathlib import Path

from src.model_registry import ModelRegistry, ModelVersion


class TestModelRegistry:
    
    def test_save_sklearn_model(self, registry, mock_sklearn_model):
        """Test saving an sklearn model."""
        ver = registry.save("test_model", mock_sklearn_model, metrics={"accuracy": 0.95})
        
        assert ver.version == "1.0.0"
        assert ver.model_type == "sklearn"
        assert ver.metrics["accuracy"] == 0.95
        assert (registry.models_dir / ver.filename).exists()
    
    def test_save_pytorch_model(self, registry, mock_pytorch_model):
        """Test saving a PyTorch model."""
        ver = registry.save("test_nn", mock_pytorch_model, metrics={"f1_score": 0.92})
        
        assert ver.version == "1.0.0"
        assert ver.model_type == "pytorch"
        assert ver.filename.endswith(".pt")
        assert (registry.models_dir / ver.filename).exists()
    
    def test_load_sklearn_model(self, registry, mock_sklearn_model):
        """Test loading an sklearn model."""
        registry.save("test_model", mock_sklearn_model)
        loaded = registry.load("test_model")
        
        assert hasattr(loaded, "predict")
    
    def test_load_pytorch_model(self, registry, mock_pytorch_model):
        """Test loading a PyTorch model."""
        registry.save("test_nn", mock_pytorch_model)
        loaded = registry.load("test_nn", model_class=type(mock_pytorch_model))
        
        assert hasattr(loaded, "forward")
    
    def test_auto_versioning(self, registry, mock_sklearn_model):
        """Test automatic version incrementing."""
        v1 = registry.save("model", mock_sklearn_model)
        v2 = registry.save("model", mock_sklearn_model)
        v3 = registry.save("model", mock_sklearn_model)
        
        assert v1.version == "1.0.0"
        assert v2.version == "1.0.1"
        assert v3.version == "1.0.2"
    
    def test_explicit_version(self, registry, mock_sklearn_model):
        """Test saving with explicit version."""
        ver = registry.save("model", mock_sklearn_model, version="2.0.0")
        assert ver.version == "2.0.0"
    
    def test_list_models(self, registry, mock_sklearn_model):
        """Test listing registered models."""
        registry.save("model_a", mock_sklearn_model)
        registry.save("model_b", mock_sklearn_model)
        
        models = registry.list_models()
        assert "model_a" in models
        assert "model_b" in models
    
    def test_list_versions(self, registry, mock_sklearn_model):
        """Test listing versions of a model."""
        registry.save("model", mock_sklearn_model, version="1.0.0")
        registry.save("model", mock_sklearn_model, version="1.1.0")
        
        versions = registry.list_versions("model")
        assert "1.0.0" in versions
        assert "1.1.0" in versions
    
    def test_set_active_version(self, registry, mock_sklearn_model):
        """Test rollback to previous version."""
        registry.save("model", mock_sklearn_model, version="1.0.0")
        registry.save("model", mock_sklearn_model, version="2.0.0")
        
        assert registry.get_active_version("model") == "2.0.0"
        
        registry.set_active("model", "1.0.0")
        assert registry.get_active_version("model") == "1.0.0"
    
    def test_set_active_invalid_version(self, registry, mock_sklearn_model):
        """Test rollback to non-existent version raises error."""
        registry.save("model", mock_sklearn_model)
        
        with pytest.raises(ValueError):
            registry.set_active("model", "9.9.9")
    
    def test_get_metrics(self, registry, mock_sklearn_model):
        """Test retrieving model metrics."""
        registry.save("model", mock_sklearn_model, metrics={"f1": 0.9, "acc": 0.95})
        
        metrics = registry.get_metrics("model")
        assert metrics["f1"] == 0.9
        assert metrics["acc"] == 0.95
    
    def test_delete_version(self, registry, mock_sklearn_model):
        """Test deleting a model version."""
        registry.save("model", mock_sklearn_model, version="1.0.0")
        registry.save("model", mock_sklearn_model, version="2.0.0")
        
        registry.delete_version("model", "1.0.0")
        
        assert "1.0.0" not in registry.list_versions("model")
        assert "2.0.0" in registry.list_versions("model")
    
    def test_manifest_persistence(self, temp_models_dir, mock_sklearn_model):
        """Test that manifest persists across registry instances."""
        reg1 = ModelRegistry(str(temp_models_dir))
        reg1.save("model", mock_sklearn_model, metrics={"acc": 0.9})
        
        reg2 = ModelRegistry(str(temp_models_dir))
        assert "model" in reg2.list_models()
        assert reg2.get_metrics("model")["acc"] == 0.9
    
    def test_cache_clearing(self, registry, mock_sklearn_model):
        """Test cache clearing."""
        registry.save("model", mock_sklearn_model)
        registry.load("model")
        
        assert len(registry._cache) > 0
        registry.clear_cache()
        assert len(registry._cache) == 0


class TestModelVersion:
    
    def test_is_better_than(self):
        """Test version comparison by metric."""
        v1 = ModelVersion("1.0.0", "sklearn", "m.pkl", "", {"f1_score": 0.8}, {})
        v2 = ModelVersion("2.0.0", "sklearn", "m.pkl", "", {"f1_score": 0.9}, {})
        
        assert v2.is_better_than(v1)
        assert not v1.is_better_than(v2)
    
    def test_is_better_than_custom_metric(self):
        """Test version comparison with custom metric."""
        v1 = ModelVersion("1.0.0", "sklearn", "m.pkl", "", {"accuracy": 0.95}, {})
        v2 = ModelVersion("2.0.0", "sklearn", "m.pkl", "", {"accuracy": 0.90}, {})
        
        assert v1.is_better_than(v2, metric="accuracy")
