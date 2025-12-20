"""Test script for HybridPredictor."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_predictor import create_predictor


def test_payload_detection():
    """Test payload classification."""
    print("\n=== Payload Detection Test ===")
    
    payloads = [
        "SELECT * FROM users WHERE id=1 OR 1=1--",
        "<script>alert('XSS')</script>",
        "hello world",
        "john.doe@example.com",
        "'; DROP TABLE users;--",
        "normal search query",
    ]
    
    results = predictor.predict({'payloads': payloads})
    
    for payload, prob, pred in zip(payloads, results['scores']['payload'], results['is_attack']):
        status = "🚨 MALICIOUS" if pred else "✓ Safe"
        print(f"  [{prob:.2%}] {status}: {payload[:50]}")


def test_url_detection():
    """Test URL classification."""
    print("\n=== URL Detection Test ===")
    
    urls = [
        "https://www.google.com/search?q=test",
        "http://paypa1-secure.tk/login",
        "https://github.com/user/repo",
        "http://192.168.1.1/admin/malware.exe",
        "https://amazon.com/products",
        "http://free-iphone.ml/claim",
    ]
    
    results = predictor.predict({'urls': urls})
    
    for url, prob, pred in zip(urls, results['scores']['url'], results['is_attack']):
        status = "🚨 MALICIOUS" if pred else "✓ Safe"
        print(f"  [{prob:.2%}] {status}: {url[:60]}")


def test_timeseries_detection():
    """Test time-series anomaly detection."""
    print("\n=== Time-Series Detection Test ===")
    
    # Normal traffic pattern
    normal = np.random.randn(60, 8) * 0.1 + 0.5
    
    # Attack pattern (spike)
    attack = np.random.randn(60, 8) * 0.1 + 0.5
    attack[30:, 0] += 2.0  # Spike in first feature
    
    sequences = np.stack([normal, attack])
    results = predictor.predict({'timeseries': sequences})
    
    labels = ['Normal traffic', 'Attack traffic']
    for label, prob, pred in zip(labels, results['scores']['timeseries'], results['is_attack']):
        status = "🚨 ANOMALY" if pred else "✓ Normal"
        print(f"  [{prob:.2%}] {status}: {label}")


def test_combined():
    """Test combined prediction."""
    print("\n=== Combined Prediction Test ===")
    
    # Malicious scenario
    mal_result = predictor.predict({
        'payloads': ["' OR '1'='1"],
        'urls': ["http://evil.tk/phish"],
    })
    
    # Benign scenario
    ben_result = predictor.predict({
        'payloads': ["Hello, how are you?"],
        'urls': ["https://google.com"],
    })
    
    print(f"  Malicious input:")
    print(f"    Final: {mal_result['confidence'][0]:.2%} ({'ATTACK' if mal_result['is_attack'][0] else 'SAFE'})")
    print(f"    Scores: payload={mal_result['scores']['payload'][0]:.2%}, url={mal_result['scores']['url'][0]:.2%}")
    
    print(f"  Benign input:")
    print(f"    Final: {ben_result['confidence'][0]:.2%} ({'ATTACK' if ben_result['is_attack'][0] else 'SAFE'})")
    print(f"    Scores: payload={ben_result['scores']['payload'][0]:.2%}, url={ben_result['scores']['url'][0]:.2%}")


if __name__ == "__main__":
    print("=" * 50)
    print("HybridPredictor Test Suite")
    print("=" * 50)
    
    # Load predictor
    models_dir = Path(__file__).parent.parent / 'models'
    predictor = create_predictor(str(models_dir))
    
    # Run tests
    test_payload_detection()
    test_url_detection()
    test_timeseries_detection()
    test_combined()
    
    print("\n" + "=" * 50)
    print("Tests Complete!")
    print("=" * 50)
