"""Stress test runner - orchestrates all test suites."""
import sys
import time
import random
import hashlib
from datetime import date
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.stress_test.generators.adversarial import AdversarialGenerator
from src.stress_test.generators.edge_cases import EdgeCaseGenerator
from src.stress_test.generators.scrapers import get_fresh_samples, FallbackGenerator
from src.stress_test.suites.performance_suite import PerformanceSuite
from src.stress_test.metrics import MetricsCalculator, MetricsResult
from src.stress_test.alerting import DiscordAlerter
from src.stress_test.reporter import ReportGenerator, StressTestReport
from src.stress_test.hash_registry import HashRegistry


@dataclass
class StressTestConfig:
    """Configuration for stress test."""
    # Sample counts
    adversarial_benign_count: int = 7500
    adversarial_malicious_count: int = 2500
    scraped_benign_count: int = 7500
    edge_case_count: int = 2500
    
    # Thresholds
    accuracy_threshold: float = 0.99
    fp_threshold: float = 0.03
    fn_threshold: float = 0.02
    latency_threshold_ms: float = 100.0
    calibration_threshold: float = 0.1
    
    # Paths
    model_path: str = "models/payload_cnn.pt"
    hash_registry_path: str = "models/training_hashes.pkl"
    reports_dir: str = "reports"
    
    # Discord (default webhook - always used)
    discord_webhook: str = "https://discord.com/api/webhooks/1452715933398466782/Ajftu5_fHelFqifTRcZN3S7fCDddXPs89p9w8dTHX8pF1xUO59ckac_DyCTQsRKC1H8O"
    
    # Options
    verify_no_overlap: bool = True
    run_performance_tests: bool = True
    send_discord: bool = True


class StressTestRunner:
    """Main stress test orchestrator."""
    
    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()
        
        # Initialize components
        self.seed = int(hashlib.md5(date.today().isoformat().encode()).hexdigest()[:8], 16)
        random.seed(self.seed)
        
        self.adversarial_gen = AdversarialGenerator(seed=self.seed)
        self.edge_case_gen = EdgeCaseGenerator(seed=self.seed)
        self.metrics_calc = MetricsCalculator(
            accuracy_threshold=self.config.accuracy_threshold,
            fp_threshold=self.config.fp_threshold,
            fn_threshold=self.config.fn_threshold,
            calibration_threshold=self.config.calibration_threshold,
        )
        self.performance_suite = PerformanceSuite(
            latency_threshold_ms=self.config.latency_threshold_ms,
        )
        self.reporter = ReportGenerator(Path(self.config.reports_dir))
        self.alerter = DiscordAlerter(self.config.discord_webhook)
        
        # Hash registry for overlap checking
        self.hash_registry = HashRegistry(Path(self.config.hash_registry_path))
        
        # Model (to be loaded)
        self.model = None
        self.predict_fn = None
        self.predict_batch_fn = None
    
    def load_model(self):
        """Load the model for testing."""
        import torch
        from src.torch_models.payload_cnn import PayloadCNN
        
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PayloadCNN().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # Create prediction functions
        def predict(text: str) -> Tuple[float, int]:
            encoded = [ord(c) % 256 for c in text[:512]]
            encoded += [0] * (512 - len(encoded))
            x = torch.tensor([encoded], dtype=torch.long, device=device)
            with torch.no_grad():
                prob = torch.sigmoid(self.model(x)).item()
            return prob, 1 if prob > 0.5 else 0
        
        def predict_batch(texts: List[str]) -> List[Tuple[float, int]]:
            encoded = []
            for text in texts:
                enc = [ord(c) % 256 for c in text[:512]]
                enc += [0] * (512 - len(enc))
                encoded.append(enc)
            x = torch.tensor(encoded, dtype=torch.long, device=device)
            with torch.no_grad():
                probs = torch.sigmoid(self.model(x)).squeeze().tolist()
            if isinstance(probs, float):
                probs = [probs]
            return [(p, 1 if p > 0.5 else 0) for p in probs]
        
        self.predict_fn = predict
        self.predict_batch_fn = predict_batch
        
        print(f"Model loaded from {model_path}")
        return self
    
    def generate_test_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Generate 50/50 adversarial + scraped test data."""
        print("Generating test data...")
        
        samples = []
        labels = []
        categories = []
        
        # 50% Adversarial generated
        print("  Generating adversarial samples...")
        
        # Benign adversarial (samples that look suspicious but are benign)
        for text in self.adversarial_gen.generate_benign_adversarial(self.config.adversarial_benign_count):
            samples.append(text)
            labels.append(0)
            categories.append('adversarial_benign')
        
        # Edge cases (benign)
        edge_cases = self.edge_case_gen.generate_flat(self.config.edge_case_count)
        for text in edge_cases:
            samples.append(text)
            labels.append(0)
            categories.append('edge_case')
        
        # Malicious adversarial (obfuscated attacks)
        malicious_payloads = [
            "' OR '1'='1",
            "<script>alert(1)</script>",
            "'; DROP TABLE users; --",
            "{{7*7}}",
            "${7*7}",
            "| cat /etc/passwd",
            "../../../etc/passwd",
        ]
        
        for _ in range(self.config.adversarial_malicious_count):
            payload = random.choice(malicious_payloads)
            variants = self.adversarial_gen.generate_from_payload(payload)
            variant = random.choice(variants)
            samples.append(variant)
            labels.append(1)
            categories.append('adversarial_malicious')
        
        # 50% Scraped real-world
        print("  Scraping fresh real-world data...")
        scraped = get_fresh_samples(
            benign_count=self.config.scraped_benign_count,
            malicious_count=500,  # CVE descriptions for context
        )
        
        for text in scraped['benign']:
            samples.append(text)
            labels.append(0)
            categories.append('scraped_benign')
        
        # Note: CVE descriptions are not actually malicious payloads,
        # they're descriptions. We use them for context testing only.
        
        print(f"  Generated {len(samples)} total samples")
        print(f"    Adversarial benign: {self.config.adversarial_benign_count}")
        print(f"    Edge cases: {self.config.edge_case_count}")
        print(f"    Adversarial malicious: {self.config.adversarial_malicious_count}")
        print(f"    Scraped benign: {len(scraped['benign'])}")
        
        return samples, labels, categories
    
    def verify_no_overlap(self, samples: List[str]) -> Tuple[int, int]:
        """Verify test samples don't overlap with training data."""
        if not self.config.verify_no_overlap:
            return 0, len(samples)
        
        print("Verifying no overlap with training data...")
        
        if not self.hash_registry.load():
            print("  Warning: Hash registry not found, skipping overlap check")
            return 0, len(samples)
        
        overlaps, total = self.hash_registry.verify_no_overlap(iter(samples))
        
        if overlaps > 0:
            print(f"  ⚠️ WARNING: {overlaps}/{total} samples overlap with training data!")
        else:
            print(f"  ✅ No overlap detected ({total} samples checked)")
        
        return overlaps, total
    
    def run_inference(self, samples: List[str], labels: List[int]) -> Tuple[List[int], List[float]]:
        """Run model inference on all samples."""
        print("Running inference...")
        
        predictions = []
        probabilities = []
        
        batch_size = 64
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            results = self.predict_batch_fn(batch)
            for prob, pred in results:
                probabilities.append(prob)
                predictions.append(pred)
            
            if (i + batch_size) % 1000 == 0:
                print(f"  Processed {min(i + batch_size, len(samples))}/{len(samples)}")
        
        return predictions, probabilities
    
    def collect_failed_samples(self,
                                samples: List[str],
                                labels: List[int],
                                predictions: List[int],
                                probabilities: List[float],
                                max_samples: int = 20) -> List[Dict]:
        """Collect samples that were misclassified."""
        failed = []
        
        for i, (text, label, pred, prob) in enumerate(zip(samples, labels, predictions, probabilities)):
            if label != pred:
                failed.append({
                    'text': text[:100],
                    'label': 'FP' if label == 0 else 'FN',
                    'true_label': label,
                    'predicted': pred,
                    'score': prob,
                })
                
                if len(failed) >= max_samples:
                    break
        
        return failed
    
    def run(self) -> StressTestReport:
        """Run complete stress test."""
        start_time = time.time()
        print("=" * 60)
        print("         DAILY STRESS TEST")
        print("=" * 60)
        print(f"Date: {date.today().isoformat()}")
        print(f"Seed: {self.seed}")
        print()
        
        # Load model
        self.load_model()
        
        # Generate test data
        samples, labels, categories = self.generate_test_data()
        
        # Verify no overlap
        overlaps, _ = self.verify_no_overlap(samples)
        
        # Run inference
        predictions, probabilities = self.run_inference(samples, labels)
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.metrics_calc.calculate(labels, predictions, probabilities, categories)
        print(self.metrics_calc.format_metrics(metrics))
        
        # Run performance tests
        performance_results = {}
        if self.config.run_performance_tests:
            print("\nRunning performance tests...")
            perf_samples = samples[:500]
            
            latency = self.performance_suite.test_inference_latency(
                lambda t: self.predict_fn(t),
                perf_samples,
            )
            
            throughput = self.performance_suite.test_batch_throughput(
                lambda batch: self.predict_batch_fn(batch),
                samples[:1000],
            )
            
            performance_results = {
                'latency': {
                    'p50_ms': latency.p50_ms,
                    'p95_ms': latency.p95_ms,
                    'p99_ms': latency.p99_ms,
                    'samples_per_sec': latency.samples_per_sec,
                },
                'throughput': {bs: r.samples_per_sec for bs, r in throughput.items()},
                'memory': {'peak_mb': 0},  # Would need tracemalloc
            }
            
            print(f"  P95 Latency: {latency.p95_ms:.1f}ms")
            print(f"  Throughput: {latency.samples_per_sec:.0f} samples/sec")
        
        # Collect failed samples
        failed_samples = self.collect_failed_samples(samples, labels, predictions, probabilities)
        
        # Generate report
        print("\nGenerating report...")
        test_info = {
            'model_name': 'payload_cnn',
            'model_path': self.config.model_path,
            'seed': self.seed,
            'benign_count': sum(1 for l in labels if l == 0),
            'malicious_count': sum(1 for l in labels if l == 1),
            'adversarial_count': self.config.adversarial_benign_count + self.config.adversarial_malicious_count,
            'scraped_count': self.config.scraped_benign_count,
        }
        
        report = self.reporter.generate_and_save(
            metrics={
                'accuracy': metrics.accuracy,
                'fp_rate': metrics.fp_rate,
                'fn_rate': metrics.fn_rate,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'calibration_error': metrics.calibration_error,
                'total_samples': metrics.total_samples,
                'true_positives': metrics.true_positives,
                'true_negatives': metrics.true_negatives,
                'false_positives': metrics.false_positives,
                'false_negatives': metrics.false_negatives,
                'category_metrics': metrics.category_metrics,
            },
            performance=performance_results,
            test_info=test_info,
            failed_samples=failed_samples,
        )
        
        # Send Discord alert
        if self.config.send_discord and self.config.discord_webhook:
            print("\nSending Discord notification...")
            discord_metrics = {
                'accuracy': metrics.accuracy,
                'fp_rate': metrics.fp_rate,
                'fn_rate': metrics.fn_rate,
                'p95_latency_ms': performance_results.get('latency', {}).get('p95_ms', 0),
                'calibration_error': metrics.calibration_error,
            }
            self.alerter.send_report(
                metrics=discord_metrics,
                categories=metrics.category_metrics,
                trend=report.historical_fp_rates,
                failed_samples=failed_samples,
                model_name='payload_cnn',
            )
        
        elapsed = time.time() - start_time
        print(f"\nStress test completed in {elapsed:.1f}s")
        print(f"Status: {'✅ PASSED' if report.all_passed else '❌ FAILED'}")
        
        return report


def run_stress_test(config: StressTestConfig = None) -> StressTestReport:
    """Convenience function to run stress test."""
    runner = StressTestRunner(config)
    return runner.run()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run daily stress test")
    parser.add_argument("--model", default="models/payload_cnn.pt", help="Model path")
    parser.add_argument("--webhook", help="Discord webhook URL")
    parser.add_argument("--no-discord", action="store_true", help="Skip Discord notification")
    parser.add_argument("--no-perf", action="store_true", help="Skip performance tests")
    args = parser.parse_args()
    
    config = StressTestConfig(
        model_path=args.model,
        discord_webhook=args.webhook,
        send_discord=not args.no_discord,
        run_performance_tests=not args.no_perf,
    )
    
    report = run_stress_test(config)
    
    sys.exit(0 if report.all_passed else 1)
