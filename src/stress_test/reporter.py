"""Report generation for stress test results."""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class StressTestReport:
    """Complete stress test report."""
    date: str
    model_name: str
    model_path: str
    seed: int
    
    # Metrics
    accuracy: float
    fp_rate: float
    fn_rate: float
    precision: float
    recall: float
    f1_score: float
    calibration_error: float
    
    # Performance
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    samples_per_sec: float
    peak_memory_mb: float
    
    # Counts
    total_samples: int
    benign_samples: int
    malicious_samples: int
    adversarial_samples: int
    scraped_samples: int
    
    # Results
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    
    # Status
    all_passed: bool
    failures: List[str]
    warnings: List[str]
    
    # Breakdown
    category_metrics: Dict[str, Dict[str, float]]
    
    # Failed samples
    failed_samples: List[Dict[str, Any]]
    
    # Trend
    historical_fp_rates: List[float]
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class ReportGenerator:
    """Generate JSON and human-readable reports."""
    
    def __init__(self, reports_dir: Path = None):
        self.reports_dir = Path(reports_dir) if reports_dir else Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def create_report(self,
                      metrics: Dict[str, Any],
                      performance: Dict[str, Any],
                      test_info: Dict[str, Any],
                      failed_samples: List[Dict] = None,
                      historical: List[float] = None) -> StressTestReport:
        """Create a complete report from test results."""
        
        # Determine failures and warnings
        failures = []
        warnings = []
        
        acc = metrics.get('accuracy', 0)
        fp = metrics.get('fp_rate', 0)
        fn = metrics.get('fn_rate', 0)
        
        if acc < 0.99:
            failures.append(f"Accuracy {acc*100:.2f}% < 99%")
        elif acc < 0.995:
            warnings.append(f"Accuracy approaching threshold: {acc*100:.2f}%")
        
        if fp > 0.03:
            failures.append(f"FP Rate {fp*100:.2f}% > 3%")
        elif fp > 0.025:
            warnings.append(f"FP Rate approaching threshold: {fp*100:.2f}%")
        
        if fn > 0.02:
            failures.append(f"FN Rate {fn*100:.2f}% > 2%")
        elif fn > 0.015:
            warnings.append(f"FN Rate approaching threshold: {fn*100:.2f}%")
        
        lat = performance.get('latency', {})
        if lat.get('p95_ms', 0) > 100:
            failures.append(f"P95 Latency {lat.get('p95_ms', 0):.0f}ms > 100ms")
        
        return StressTestReport(
            date=datetime.now().strftime('%Y-%m-%d'),
            model_name=test_info.get('model_name', 'payload_cnn'),
            model_path=test_info.get('model_path', ''),
            seed=test_info.get('seed', 0),
            
            accuracy=metrics.get('accuracy', 0),
            fp_rate=metrics.get('fp_rate', 0),
            fn_rate=metrics.get('fn_rate', 0),
            precision=metrics.get('precision', 0),
            recall=metrics.get('recall', 0),
            f1_score=metrics.get('f1_score', 0),
            calibration_error=metrics.get('calibration_error', 0),
            
            p50_latency_ms=lat.get('p50_ms', 0),
            p95_latency_ms=lat.get('p95_ms', 0),
            p99_latency_ms=lat.get('p99_ms', 0),
            samples_per_sec=lat.get('samples_per_sec', 0),
            peak_memory_mb=performance.get('memory', {}).get('peak_mb', 0),
            
            total_samples=metrics.get('total_samples', 0),
            benign_samples=test_info.get('benign_count', 0),
            malicious_samples=test_info.get('malicious_count', 0),
            adversarial_samples=test_info.get('adversarial_count', 0),
            scraped_samples=test_info.get('scraped_count', 0),
            
            true_positives=metrics.get('true_positives', 0),
            true_negatives=metrics.get('true_negatives', 0),
            false_positives=metrics.get('false_positives', 0),
            false_negatives=metrics.get('false_negatives', 0),
            
            all_passed=len(failures) == 0,
            failures=failures,
            warnings=warnings,
            
            category_metrics=metrics.get('category_metrics', {}),
            failed_samples=failed_samples or [],
            historical_fp_rates=historical or [],
        )
    
    def save_json(self, report: StressTestReport) -> Path:
        """Save report as JSON."""
        filename = f"stress_test_{report.date}.json"
        path = self.reports_dir / filename
        
        with open(path, 'w') as f:
            f.write(report.to_json())
        
        return path
    
    def format_human_readable(self, report: StressTestReport) -> str:
        """Format report as human-readable text."""
        lines = []
        
        # Header
        lines.append("=" * 60)
        lines.append("           DAILY STRESS TEST REPORT")
        lines.append("=" * 60)
        lines.append(f"Date: {report.date}")
        lines.append(f"Model: {report.model_name}")
        lines.append(f"Seed: {report.seed}")
        lines.append("")
        
        # Status
        if report.all_passed:
            lines.append("STATUS: ✅ ALL TESTS PASSED")
        else:
            lines.append("STATUS: ❌ TESTS FAILED")
            for f in report.failures:
                lines.append(f"  • {f}")
        
        if report.warnings:
            lines.append("\nWARNINGS:")
            for w in report.warnings:
                lines.append(f"  ⚠️ {w}")
        
        lines.append("")
        
        # Metrics table
        lines.append("METRICS:")
        lines.append("┌────────────────┬─────────┬────────┬────────┐")
        lines.append("│ Metric         │ Value   │ Target │ Status │")
        lines.append("├────────────────┼─────────┼────────┼────────┤")
        
        acc_s = "✅" if report.accuracy >= 0.99 else "❌"
        lines.append(f"│ Accuracy       │ {report.accuracy*100:5.2f}%  │ >99%   │   {acc_s}   │")
        
        fp_s = "✅" if report.fp_rate <= 0.03 else "❌"
        lines.append(f"│ FP Rate        │ {report.fp_rate*100:5.2f}%  │ <3%    │   {fp_s}   │")
        
        fn_s = "✅" if report.fn_rate <= 0.02 else "❌"
        lines.append(f"│ FN Rate        │ {report.fn_rate*100:5.2f}%  │ <2%    │   {fn_s}   │")
        
        lat_s = "✅" if report.p95_latency_ms <= 100 else "❌"
        lines.append(f"│ P95 Latency    │ {report.p95_latency_ms:5.0f}ms │ <100ms │   {lat_s}   │")
        
        cal_s = "✅" if report.calibration_error <= 0.1 else "❌"
        lines.append(f"│ Calibration    │ {report.calibration_error:5.3f}   │ <0.1   │   {cal_s}   │")
        
        lines.append("└────────────────┴─────────┴────────┴────────┘")
        lines.append("")
        
        # Performance
        lines.append("PERFORMANCE:")
        lines.append(f"  Latency: P50={report.p50_latency_ms:.1f}ms, P95={report.p95_latency_ms:.1f}ms, P99={report.p99_latency_ms:.1f}ms")
        lines.append(f"  Throughput: {report.samples_per_sec:.0f} samples/sec")
        lines.append(f"  Peak Memory: {report.peak_memory_mb:.1f} MB")
        lines.append("")
        
        # Test coverage
        lines.append("TEST COVERAGE:")
        lines.append(f"  Total Samples: {report.total_samples:,}")
        lines.append(f"  Benign: {report.benign_samples:,} | Malicious: {report.malicious_samples:,}")
        lines.append(f"  Adversarial: {report.adversarial_samples:,} | Scraped: {report.scraped_samples:,}")
        lines.append("")
        
        # Confusion matrix
        lines.append("CONFUSION MATRIX:")
        lines.append(f"  TP: {report.true_positives:,}  |  FP: {report.false_positives:,}")
        lines.append(f"  FN: {report.false_negatives:,}  |  TN: {report.true_negatives:,}")
        lines.append("")
        
        # Category breakdown
        if report.category_metrics:
            lines.append("CATEGORY BREAKDOWN:")
            for cat, m in sorted(report.category_metrics.items()):
                fp_warn = "⚠️" if m.get('fp_rate', 0) > 0.025 else ""
                lines.append(f"  {cat:12s}: {m.get('accuracy', 0)*100:5.1f}% acc, {m.get('fp_rate', 0)*100:4.1f}% FP {fp_warn}")
            lines.append("")
        
        # Trend
        if report.historical_fp_rates:
            trend = " → ".join([f"{r*100:.1f}%" for r in report.historical_fp_rates[-7:]])
            lines.append(f"7-DAY FP TREND: {trend}")
            lines.append("")
        
        # Failed samples
        if report.failed_samples:
            lines.append("FAILED SAMPLES (top 5):")
            for i, s in enumerate(report.failed_samples[:5], 1):
                text = s.get('text', '')[:40]
                score = s.get('score', 0)
                label = s.get('label', 'FP')
                lines.append(f"  {i}. \"{text}...\" → {score:.2f} ({label})")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_human_readable(self, report: StressTestReport) -> Path:
        """Save human-readable report."""
        filename = f"stress_test_{report.date}.txt"
        path = self.reports_dir / filename
        
        with open(path, 'w') as f:
            f.write(self.format_human_readable(report))
        
        return path
    
    def load_historical(self, days: int = 7) -> List[float]:
        """Load historical FP rates from past reports."""
        fp_rates = []
        
        for report_file in sorted(self.reports_dir.glob("stress_test_*.json"))[-days:]:
            try:
                with open(report_file) as f:
                    data = json.load(f)
                    fp_rates.append(data.get('fp_rate', 0))
            except:
                continue
        
        return fp_rates
    
    def generate_and_save(self,
                          metrics: Dict[str, Any],
                          performance: Dict[str, Any],
                          test_info: Dict[str, Any],
                          failed_samples: List[Dict] = None) -> StressTestReport:
        """Generate report and save both formats."""
        historical = self.load_historical()
        
        report = self.create_report(
            metrics=metrics,
            performance=performance,
            test_info=test_info,
            failed_samples=failed_samples,
            historical=historical,
        )
        
        json_path = self.save_json(report)
        txt_path = self.save_human_readable(report)
        
        print(f"Reports saved:")
        print(f"  JSON: {json_path}")
        print(f"  Text: {txt_path}")
        
        return report


if __name__ == "__main__":
    # Test report generation
    gen = ReportGenerator(Path("reports"))
    
    metrics = {
        'accuracy': 0.9987,
        'fp_rate': 0.012,
        'fn_rate': 0.008,
        'precision': 0.988,
        'recall': 0.992,
        'f1_score': 0.990,
        'calibration_error': 0.04,
        'total_samples': 30000,
        'true_positives': 4960,
        'true_negatives': 24800,
        'false_positives': 300,
        'false_negatives': 40,
        'category_metrics': {
            'email': {'accuracy': 0.998, 'fp_rate': 0.01, 'count': 5000},
            'code': {'accuracy': 0.995, 'fp_rate': 0.02, 'count': 5000},
            'unicode': {'accuracy': 0.989, 'fp_rate': 0.028, 'count': 5000},
        },
    }
    
    performance = {
        'latency': {'p50_ms': 25, 'p95_ms': 45, 'p99_ms': 78, 'samples_per_sec': 1200},
        'memory': {'peak_mb': 450},
    }
    
    test_info = {
        'model_name': 'payload_cnn_v2',
        'model_path': 'models/payload_cnn.pt',
        'seed': 12345,
        'benign_count': 25000,
        'malicious_count': 5000,
        'adversarial_count': 15000,
        'scraped_count': 15000,
    }
    
    report = gen.create_report(metrics, performance, test_info)
    print(gen.format_human_readable(report))
