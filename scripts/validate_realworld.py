"""Phase E2: Real-world validation test suite."""
import sys
from pathlib import Path
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# Real-world test cases from CTF, bug bounties, and security research
PAYLOAD_TEST_CASES = {
    'sql_injection': [
        ("' OR '1'='1", True),
        ("1' AND '1'='1", True),
        ("admin'--", True),
        ("1; DROP TABLE users--", True),
        ("' UNION SELECT * FROM users--", True),
        ("1' ORDER BY 1--", True),
    ],
    'xss': [
        ("<script>alert('XSS')</script>", True),
        ("<img src=x onerror=alert(1)>", True),
        ("<svg onload=alert(1)>", True),
        ("javascript:alert(1)", True),
        ("<body onload=alert(1)>", True),
    ],
    'command_injection': [
        ("; ls -la", True),
        ("| cat /etc/passwd", True),
        ("$(whoami)", True),
        ("`id`", True),
        ("& ping -c 10 127.0.0.1", True),
    ],
    'template_injection': [
        ("{{7*7}}", True),
        ("${7*7}", True),
        ("#{7*7}", True),
        ("<%= 7*7 %>", True),
    ],
    'benign_edge_cases': [
        ("john.doe@example.com", False),
        ("O'Brien", False),  # Irish name with apostrophe
        ("SELECT * FROM menu", False),  # Menu item
        ("The script was great!", False),
        ("Price: $100 OR best offer", False),
        ("1+1=2", False),
        ("C:\\Users\\Admin", False),
        ("<3 love this product", False),
    ]
}

URL_TEST_CASES = {
    'phishing': [
        ("http://paypa1.com/login", True),  # Typosquat
        ("http://arnazon.com/signin", True),
        ("http://secure-paypal.tk/verify", True),
        ("http://paypal.account-verify.com/", True),
        ("http://192.168.1.1/paypal/login", True),
    ],
    'malware': [
        ("http://download123.xyz/file.exe", True),
        ("http://free-software.ml/crack.zip", True),
        ("http://update-flash.pw/player.exe", True),
    ],
    'benign_edge_cases': [
        ("https://www.google.com/search?q=paypal+login", False),
        ("https://krebsonsecurity.com/malware-analysis", False),  # Security blog
        ("https://github.com/paypal/sdk", False),
        ("https://stackoverflow.com/questions/sql-injection", False),
        ("https://en.wikipedia.org/wiki/Phishing", False),
        ("https://api.stripe.com/v1/charges", False),
    ]
}


def run_payload_tests(predictor):
    """Run payload test cases."""
    print("\n" + "="*60)
    print(" PAYLOAD VALIDATION TESTS")
    print("="*60)
    
    results = {'passed': 0, 'failed': 0, 'failures': []}
    
    for category, cases in PAYLOAD_TEST_CASES.items():
        print(f"\n[{category.upper()}]")
        for payload, expected_malicious in cases:
            prob = predictor.predict_payload([payload])[0]
            predicted = prob > 0.5
            passed = predicted == expected_malicious
            
            status = "✓" if passed else "✗"
            results['passed' if passed else 'failed'] += 1
            
            if not passed:
                results['failures'].append({
                    'category': category,
                    'input': payload,
                    'expected': bool(expected_malicious),
                    'predicted': bool(predicted),
                    'confidence': float(prob)
                })
            
            print(f"  {status} '{payload[:40]}...' -> {prob:.1%} (expected: {'MAL' if expected_malicious else 'BEN'})")
    
    return results


def run_url_tests(predictor):
    """Run URL test cases."""
    print("\n" + "="*60)
    print(" URL VALIDATION TESTS")
    print("="*60)
    
    results = {'passed': 0, 'failed': 0, 'failures': []}
    
    for category, cases in URL_TEST_CASES.items():
        print(f"\n[{category.upper()}]")
        for url, expected_malicious in cases:
            prob = predictor.predict_url([url])[0]
            predicted = prob > 0.5
            passed = predicted == expected_malicious
            
            status = "✓" if passed else "✗"
            results['passed' if passed else 'failed'] += 1
            
            if not passed:
                results['failures'].append({
                    'category': category,
                    'input': url,
                    'expected': bool(expected_malicious),
                    'predicted': bool(predicted),
                    'confidence': float(prob)
                })
            
            print(f"  {status} '{url[:50]}' -> {prob:.1%} (expected: {'MAL' if expected_malicious else 'BEN'})")
    
    return results


def run_validation_suite(models_dir='models'):
    """Run complete validation suite."""
    from hybrid_predictor import HybridPredictor
    
    print("Loading models...")
    predictor = HybridPredictor(models_dir)
    predictor.load_models()
    
    all_results = {}
    
    if 'payload_cnn' in predictor.pytorch_models:
        all_results['payload'] = run_payload_tests(predictor)
    
    if 'url_cnn' in predictor.pytorch_models:
        all_results['url'] = run_url_tests(predictor)
    
    # Summary
    print("\n" + "="*60)
    print(" VALIDATION SUMMARY")
    print("="*60)
    
    total_passed = sum(r['passed'] for r in all_results.values())
    total_failed = sum(r['failed'] for r in all_results.values())
    total = total_passed + total_failed
    
    for model, results in all_results.items():
        rate = results['passed'] / (results['passed'] + results['failed']) * 100
        print(f"  {model.upper()}: {results['passed']}/{results['passed']+results['failed']} passed ({rate:.1f}%)")
    
    print(f"\n  TOTAL: {total_passed}/{total} passed ({total_passed/total*100:.1f}%)")
    
    # Save report
    output_dir = Path(models_dir).parent / 'evaluation'
    output_dir.mkdir(exist_ok=True)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_tests': total,
            'passed': total_passed,
            'failed': total_failed,
            'pass_rate': total_passed / total if total > 0 else 0
        },
        'results': all_results
    }
    
    with open(output_dir / 'validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Validation report saved to {output_dir / 'validation_report.json'}")
    
    return all_results


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    run_validation_suite(base_path / 'models')
