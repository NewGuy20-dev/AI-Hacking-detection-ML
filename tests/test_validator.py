"""Tests for InputValidator."""
import numpy as np
import pytest

from src.input_validator import InputValidator, ValidationError


class TestPayloadValidation:
    
    def test_valid_payload(self):
        """Test valid payload passes."""
        result = InputValidator.validate_payload("SELECT * FROM users")
        assert result == "SELECT * FROM users"
    
    def test_payload_truncation(self):
        """Test long payload is truncated."""
        long_payload = "x" * 1000
        result = InputValidator.validate_payload(long_payload)
        assert len(result) == 500
    
    def test_payload_no_truncation(self):
        """Test truncation can be disabled."""
        long_payload = "x" * 600
        result = InputValidator.validate_payload(long_payload, truncate=False)
        assert len(result) == 600
    
    def test_payload_too_long(self):
        """Test payload exceeding max length raises error."""
        huge_payload = "x" * 15000
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_payload(huge_payload)
    
    def test_payload_invalid_type(self):
        """Test non-string payload raises error."""
        with pytest.raises(ValidationError, match="must be string"):
            InputValidator.validate_payload(12345)
        
        with pytest.raises(ValidationError, match="must be string"):
            InputValidator.validate_payload(["list", "of", "strings"])
    
    def test_payload_null_bytes_removed(self):
        """Test null bytes are stripped."""
        result = InputValidator.validate_payload("hello\x00world")
        assert "\x00" not in result
        assert result == "helloworld"


class TestURLValidation:
    
    def test_valid_url_with_scheme(self):
        """Test URL with scheme passes."""
        result = InputValidator.validate_url("https://example.com")
        assert result == "https://example.com"
    
    def test_url_adds_scheme(self):
        """Test scheme is added if missing."""
        result = InputValidator.validate_url("example.com/path")
        assert result == "http://example.com/path"
    
    def test_url_truncation(self):
        """Test long URL is truncated."""
        long_url = "https://example.com/" + "x" * 300
        result = InputValidator.validate_url(long_url)
        assert len(result) == 200
    
    def test_url_too_long(self):
        """Test URL exceeding max length raises error."""
        huge_url = "https://example.com/" + "x" * 3000
        with pytest.raises(ValidationError, match="too long"):
            InputValidator.validate_url(huge_url)
    
    def test_url_invalid_type(self):
        """Test non-string URL raises error."""
        with pytest.raises(ValidationError, match="must be string"):
            InputValidator.validate_url(None)
    
    def test_url_whitespace_stripped(self):
        """Test whitespace is stripped."""
        result = InputValidator.validate_url("  https://example.com  ")
        assert result == "https://example.com"


class TestNetworkFeaturesValidation:
    
    def test_valid_features_list(self):
        """Test valid feature list passes."""
        features = [0.1] * 41
        result = InputValidator.validate_network_features(features)
        assert result.shape == (1, 41)
    
    def test_valid_features_array(self):
        """Test valid numpy array passes."""
        features = np.random.randn(5, 41)
        result = InputValidator.validate_network_features(features)
        assert result.shape == (5, 41)
    
    def test_features_wrong_dimension(self):
        """Test wrong feature count raises error."""
        features = [0.1] * 30
        with pytest.raises(ValidationError, match="Expected 41 features"):
            InputValidator.validate_network_features(features)
    
    def test_features_custom_dimension(self):
        """Test custom expected dimension."""
        features = [0.1] * 10
        result = InputValidator.validate_network_features(features, expected_dim=10)
        assert result.shape == (1, 10)
    
    def test_features_nan_rejected(self):
        """Test NaN values raise error."""
        features = [0.1] * 40 + [float('nan')]
        with pytest.raises(ValidationError, match="NaN"):
            InputValidator.validate_network_features(features)
    
    def test_features_inf_rejected(self):
        """Test Inf values raise error."""
        features = [0.1] * 40 + [float('inf')]
        with pytest.raises(ValidationError, match="Inf"):
            InputValidator.validate_network_features(features)
    
    def test_features_non_numeric(self):
        """Test non-numeric values raise error."""
        features = ["a", "b", "c"]
        with pytest.raises(ValidationError, match="numeric"):
            InputValidator.validate_network_features(features, expected_dim=3)


class TestBatchValidation:
    
    def test_batch_all_valid(self):
        """Test batch with all valid items."""
        items = ["payload1", "payload2", "payload3"]
        result = InputValidator.validate_batch(items, InputValidator.validate_payload)
        assert len(result) == 3
    
    def test_batch_with_error_raises(self):
        """Test batch with invalid item raises by default."""
        items = ["valid", 12345, "also valid"]
        with pytest.raises(ValidationError, match="Item 1"):
            InputValidator.validate_batch(items, InputValidator.validate_payload)
    
    def test_batch_collect_errors(self):
        """Test batch collecting errors instead of raising."""
        items = ["valid", 12345, "also valid", None]
        valid, errors = InputValidator.validate_batch(
            items, InputValidator.validate_payload, collect_errors=True
        )
        
        assert len(valid) == 2
        assert len(errors) == 2
        assert errors[0][0] == 1  # Index of first error
        assert errors[1][0] == 3  # Index of second error
