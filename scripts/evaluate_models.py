"""Phase E1: Comprehensive evaluation script with metrics and visualization."""
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def calculate_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive classification metrics."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'false_positive_rate': fpr,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }
    
    # AUC if probabilities provided
    if y_prob is not None:
        y_prob = np.array(y_prob)
        # Simple AUC calculation
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]
        if len(pos_probs) > 0 and len(neg_probs) > 0:
            auc = np.mean([p > n for p in pos_probs for n in neg_probs])
            metrics['auc_roc'] = auc
    
    return metrics


def print_metrics(metrics, name="Model"):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f" {name} Evaluation Results")
    print(f"{'='*50}")
    print(f"  Accuracy:    {metrics['accuracy']:.2%}")
    print(f"  Precision:   {metrics['precision']:.2%}")
    print(f"  Recall:      {metrics['recall']:.2%}")
    print(f"  F1 Score:    {metrics['f1_score']:.2%}")
    print(f"  FP Rate:     {metrics['false_positive_rate']:.2%}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"              Pred 0    Pred 1")
    print(f"  Actual 0    {metrics['confusion_matrix'][0][0]:6d}    {metrics['confusion_matrix'][0][1]:6d}")
    print(f"  Actual 1    {metrics['confusion_matrix'][1][0]:6d}    {metrics['confusion_matrix'][1][1]:6d}")


def evaluate_payload_model(predictor, test_data):
    """Evaluate payload CNN model."""
    payloads, labels = test_data
    probs = predictor.predict_payload(payloads)
    preds = (probs > 0.5).astype(int)
    return calculate_metrics(labels, preds, probs)


def evaluate_url_model(predictor, test_data):
    """Evaluate URL CNN model."""
    urls, labels = test_data
    probs = predictor.predict_url(urls)
    preds = (probs > 0.5).astype(int)
    return calculate_metrics(labels, preds, probs)


def evaluate_timeseries_model(predictor, test_data):
    """Evaluate time-series LSTM model."""
    sequences, labels = test_data
    probs = predictor.predict_timeseries(sequences)
    preds = (probs > 0.5).astype(int)
    return calculate_metrics(labels, preds, probs)


def generate_test_data():
    """Generate test data for evaluation."""
    # Payload test data
    malicious_payloads = [
        "' OR 1=1--", "<script>alert(1)</script>", "'; DROP TABLE users;--",
        "{{7*7}}", "${7*7}", "| ls -la", "; cat /etc/passwd",
        "<img src=x onerror=alert(1)>", "admin'--", "1; SELECT * FROM users"
    ] * 50
    
    benign_payloads = [
        "Hello world", "john.doe@example.com", "The quick brown fox",
        "Order #12345", "Thank you for your purchase", "Meeting at 3pm",
        "Password123!", "New York, NY 10001", "+1-555-123-4567", "2024-01-15"
    ] * 50
    
    payload_data = (malicious_payloads + benign_payloads, 
                   [1]*len(malicious_payloads) + [0]*len(benign_payloads))
    
    # URL test data
    malicious_urls = [
        "http://paypa1.com/login", "http://192.168.1.1/malware.exe",
        "http://secure-amazon.tk/verify", "http://g00gle.com/signin",
        "http://free-iphone.ml/claim"
    ] * 100
    
    benign_urls = [
        "https://www.google.com/search", "https://github.com/user/repo",
        "https://amazon.com/products", "https://stackoverflow.com/questions",
        "https://en.wikipedia.org/wiki/Main_Page"
    ] * 100
    
    url_data = (malicious_urls + benign_urls,
               [1]*len(malicious_urls) + [0]*len(benign_urls))
    
    # Time-series test data (simplified)
    np.random.seed(42)
    normal_seq = np.random.randn(200, 60, 8) * 0.5
    attack_seq = np.random.randn(200, 60, 8) * 1.5
    attack_seq[:, 30:, 0] += 2  # Spike pattern
    
    ts_data = (np.vstack([normal_seq, attack_seq]),
              np.array([0]*200 + [1]*200))
    
    return payload_data, url_data, ts_data


def run_evaluation(models_dir='models'):
    """Run full evaluation suite."""
    from hybrid_predictor import HybridPredictor
    
    print("Loading models...")
    predictor = HybridPredictor(models_dir)
    predictor.load_models()
    
    print("Generating test data...")
    payload_data, url_data, ts_data = generate_test_data()
    
    results = {}
    
    # Evaluate each model
    if 'payload_cnn' in predictor.pytorch_models:
        results['payload_cnn'] = evaluate_payload_model(predictor, payload_data)
        print_metrics(results['payload_cnn'], "Payload CNN")
    
    if 'url_cnn' in predictor.pytorch_models:
        results['url_cnn'] = evaluate_url_model(predictor, url_data)
        print_metrics(results['url_cnn'], "URL CNN")
    
    if 'timeseries_lstm' in predictor.pytorch_models:
        results['timeseries_lstm'] = evaluate_timeseries_model(predictor, ts_data)
        print_metrics(results['timeseries_lstm'], "Time-Series LSTM")
    
    # Save results
    output_dir = Path(models_dir).parent / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'models_evaluated': list(results.keys()),
        'results': results
    }
    
    with open(output_dir / 'evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Evaluation report saved to {output_dir / 'evaluation_report.json'}")
    return results


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    run_evaluation(base_path / 'models')
