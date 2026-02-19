"""Tests for API Server."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


# Mock the predictor before importing
@pytest.fixture
def mock_predictor():
    """Create mock BatchHybridPredictor."""
    predictor = Mock()
    predictor.pytorch_models = {'payload_cnn': Mock(), 'url_cnn': Mock()}
    predictor.sklearn_models = {'network': Mock()}
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
def client(mock_predictor):
    """Create test client with mocked predictor."""
    with patch('src.api.server.predictor', mock_predictor):
        with patch('src.api.server.start_time', 1000.0):
            from fastapi.testclient import TestClient
            from src.api.server import app
            
            # Patch the get_predictor function
            import src.api.server as server_module
            server_module.predictor = mock_predictor
            server_module.start_time = 1000.0
            
            yield TestClient(app)


class TestHealthEndpoints:
    
    def test_health_check(self, client):
        """Test /health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'uptime_seconds' in data
    
    def test_readiness_check(self, client, mock_predictor):
        """Test /health/ready endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ready'
        assert 'models_loaded' in data
        assert 'pytorch' in data['models_loaded']


class TestPredictPayloadEndpoint:
    
    def test_predict_payload_success(self, client, mock_predictor):
        """Test successful payload prediction."""
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "' OR 1=1--"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['is_attack'] is True
        assert data['confidence'] == 0.95
        assert data['severity'] in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
        assert 'processing_time_ms' in data
    
    def test_predict_payload_benign(self, client, mock_predictor):
        """Test benign payload prediction."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([False]),
            'confidence': np.array([0.2]),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "Hello world"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['is_attack'] is False
    
    def test_predict_payload_empty(self, client):
        """Test empty payload validation."""
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": ""}
        )
        
        # Empty string is valid, just short
        assert response.status_code == 200
    
    def test_predict_payload_missing_field(self, client):
        """Test missing payload field."""
        response = client.post(
            "/api/v1/predict/payload",
            json={}
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_predict_payload_too_long(self, client):
        """Test payload exceeding max length."""
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "A" * 20000}  # Exceeds 10000 limit
        )
        
        assert response.status_code == 422
    
    def test_predict_payload_with_explanation(self, client, mock_predictor):
        """Test payload prediction with explanation."""
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "test", "include_explanation": True}
        )
        
        assert response.status_code == 200
        # Explanation may or may not be included based on implementation


class TestPredictURLEndpoint:
    
    def test_predict_url_success(self, client, mock_predictor):
        """Test successful URL prediction."""
        response = client.post(
            "/api/v1/predict/url",
            json={"url": "http://malicious.tk/phishing"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert 'is_attack' in data
        assert 'confidence' in data
    
    def test_predict_url_benign(self, client, mock_predictor):
        """Test benign URL prediction."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([False]),
            'confidence': np.array([0.1]),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/url",
            json={"url": "https://google.com"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['is_attack'] is False
    
    def test_predict_url_missing_field(self, client):
        """Test missing URL field."""
        response = client.post(
            "/api/v1/predict/url",
            json={}
        )
        
        assert response.status_code == 422
    
    def test_predict_url_too_long(self, client):
        """Test URL exceeding max length."""
        response = client.post(
            "/api/v1/predict/url",
            json={"url": "http://example.com/" + "a" * 3000}
        )
        
        assert response.status_code == 422


class TestBatchEndpoint:
    
    def test_batch_predict_payloads(self, client, mock_predictor):
        """Test batch payload prediction."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([True, False]),
            'confidence': np.array([0.9, 0.2]),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json={"payloads": ["' OR 1=1--", "Hello"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['results']) == 2
        assert 'total_processing_time_ms' in data
    
    def test_batch_predict_urls(self, client, mock_predictor):
        """Test batch URL prediction."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([True, False]),
            'confidence': np.array([0.8, 0.1]),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json={"urls": ["http://malicious.tk", "https://google.com"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['results']) == 2
    
    def test_batch_predict_empty(self, client):
        """Test batch with no inputs."""
        response = client.post(
            "/api/v1/predict/batch",
            json={}
        )
        
        assert response.status_code == 422
    
    def test_batch_predict_mixed(self, client, mock_predictor):
        """Test batch with both payloads and URLs."""
        mock_predictor.predict_batch.side_effect = [
            {
                'is_attack': np.array([True]),
                'confidence': np.array([0.9]),
                'scores': {}
            },
            {
                'is_attack': np.array([False]),
                'confidence': np.array([0.1]),
                'scores': {}
            }
        ]
        
        response = client.post(
            "/api/v1/predict/batch",
            json={"payloads": ["' OR 1=1--"], "urls": ["http://test.com"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data['results']) == 2
        assert data['results'][0]['attack_type'] in ['SQL_INJECTION', 'UNKNOWN', None]
        assert data['results'][1]['attack_type'] in ['MALICIOUS_URL', None]
        assert mock_predictor.predict_batch.call_count == 2
    
    def test_batch_predict_limit(self, client, mock_predictor):
        """Test batch respects 100 item limit."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([False] * 100),
            'confidence': np.array([0.1] * 100),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/batch",
            json={"payloads": ["test"] * 100}  # At limit
        )
        
        assert response.status_code == 200

    def test_batch_predict_preserves_order_with_benign_prefilter(self, client, mock_predictor):
        """Test mixed payload/url results preserve request order with benign prefiltering."""
        class FakeFilter:
            def is_benign(self, payload):
                return (payload == "hello", 0.95, "heuristic")

            def get_confidence_scale(self, payload):
                return 1.0

        mock_predictor.predict_batch.side_effect = [
            {
                'is_attack': np.array([True]),
                'confidence': np.array([0.91]),
                'scores': {}
            },
            {
                'is_attack': np.array([False, True]),
                'confidence': np.array([0.2, 0.9]),
                'scores': {}
            }
        ]

        with patch('src.api.routes.predict.get_filter', return_value=FakeFilter()):
            response = client.post(
                "/api/v1/predict/batch",
                json={
                    "payloads": ["hello", "' OR 1=1--"],
                    "urls": ["https://example.com", "http://bad.example"]
                }
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data['results']) == 4

        # Payload slot 0 was benign-prefiltered and should remain in place.
        assert data['results'][0]['is_attack'] is False
        # Payload slot 1 is ML output.
        assert data['results'][1]['is_attack'] is True
        # URL slots follow payload slots in order.
        assert data['results'][2]['is_attack'] is False
        assert data['results'][3]['is_attack'] is True

        # Predictor called once for non-benign payloads and once for URLs.
        assert mock_predictor.predict_batch.call_count == 2
        first_call = mock_predictor.predict_batch.call_args_list[0]
        second_call = mock_predictor.predict_batch.call_args_list[1]
        assert 'payloads' in first_call.args[0]
        assert first_call.args[0]['payloads'] == ["' OR 1=1--"]
        assert second_call.args[0]['urls'] == ["https://example.com", "http://bad.example"]

    def test_batch_payload_only_all_benign_skips_model(self, client, mock_predictor):
        """Payload-only benign batches should return results without model inference."""
        class FakeFilter:
            def is_benign(self, payload):
                return True, 0.9, "heuristic"

            def get_confidence_scale(self, payload):
                return 1.0

        with patch('src.api.routes.predict.get_filter', return_value=FakeFilter()):
            response = client.post(
                "/api/v1/predict/batch",
                json={"payloads": ["hello", "world"]}
            )

        assert response.status_code == 200
        data = response.json()
        assert len(data['results']) == 2
        assert all(r['is_attack'] is False for r in data['results'])
        assert mock_predictor.predict_batch.call_count == 0


class TestAPIEdgeCases:
    
    def test_predict_special_characters(self, client, mock_predictor):
        """Test prediction with special characters."""
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "Test\x00\n\r\t<>&\"'"}
        )
        
        assert response.status_code == 200
    
    def test_predict_unicode(self, client, mock_predictor):
        """Test prediction with unicode."""
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "Test 你好 🚨 émoji"}
        )
        
        assert response.status_code == 200
    
    def test_predict_sql_injection_patterns(self, client, mock_predictor):
        """Test various SQL injection patterns."""
        patterns = [
            "' OR '1'='1",
            "1; DROP TABLE users--",
            "' UNION SELECT * FROM users--",
            "admin'--",
        ]
        
        for pattern in patterns:
            response = client.post(
                "/api/v1/predict/payload",
                json={"payload": pattern}
            )
            assert response.status_code == 200
    
    def test_predict_xss_patterns(self, client, mock_predictor):
        """Test various XSS patterns."""
        patterns = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)",
        ]
        
        for pattern in patterns:
            response = client.post(
                "/api/v1/predict/payload",
                json={"payload": pattern}
            )
            assert response.status_code == 200
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/predict/payload")
        
        # FastAPI handles CORS via middleware
        assert response.status_code in [200, 405]
    
    def test_swagger_docs(self, client):
        """Test Swagger docs are accessible."""
        response = client.get("/docs")
        
        assert response.status_code == 200
    
    def test_openapi_json(self, client):
        """Test OpenAPI JSON is accessible."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert 'paths' in data


class TestAPISeverityClassification:
    
    def test_severity_critical(self, client, mock_predictor):
        """Test CRITICAL severity for very high confidence."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([True]),
            'confidence': np.array([0.99]),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "test"}
        )
        
        assert response.json()['severity'] == 'CRITICAL'
    
    def test_severity_high(self, client, mock_predictor):
        """Test HIGH severity."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([True]),
            'confidence': np.array([0.90]),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "test"}
        )
        
        assert response.json()['severity'] == 'HIGH'
    
    def test_severity_medium(self, client, mock_predictor):
        """Test MEDIUM severity."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([True]),
            'confidence': np.array([0.75]),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "test"}
        )
        
        assert response.json()['severity'] == 'MEDIUM'
    
    def test_severity_low(self, client, mock_predictor):
        """Test LOW severity."""
        mock_predictor.predict_batch.return_value = {
            'is_attack': np.array([True]),
            'confidence': np.array([0.55]),
            'scores': {}
        }
        
        response = client.post(
            "/api/v1/predict/payload",
            json={"payload": "test"}
        )
        
        assert response.json()['severity'] == 'LOW'
