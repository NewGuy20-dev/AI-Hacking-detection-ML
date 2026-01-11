"""Performance testing suite for model inference."""
import time
import tracemalloc
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import numpy as np


@dataclass
class PerformanceResult:
    """Results from performance tests."""
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    min_ms: float
    std_ms: float
    samples_per_sec: float
    total_samples: int
    passed: bool
    threshold_ms: float


@dataclass
class MemoryResult:
    """Results from memory tests."""
    current_mb: float
    peak_mb: float
    passed: bool
    threshold_mb: float


@dataclass
class ThroughputResult:
    """Results from throughput tests."""
    batch_size: int
    samples_per_sec: float
    batches_per_sec: float
    total_time_sec: float


class PerformanceSuite:
    """Performance testing suite for ML models."""
    
    def __init__(self, 
                 latency_threshold_ms: float = 100.0,
                 memory_threshold_mb: float = 1000.0):
        self.latency_threshold_ms = latency_threshold_ms
        self.memory_threshold_mb = memory_threshold_mb
    
    def test_inference_latency(self, 
                                predict_fn: Callable[[str], Any],
                                samples: List[str],
                                warmup: int = 10) -> PerformanceResult:
        """Test single-sample inference latency."""
        # Warmup
        for sample in samples[:warmup]:
            predict_fn(sample)
        
        # Measure
        times_ms = []
        for sample in samples:
            start = time.perf_counter()
            predict_fn(sample)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)
        
        times_arr = np.array(times_ms)
        total_time = sum(times_ms) / 1000
        
        return PerformanceResult(
            mean_ms=float(np.mean(times_arr)),
            p50_ms=float(np.percentile(times_arr, 50)),
            p95_ms=float(np.percentile(times_arr, 95)),
            p99_ms=float(np.percentile(times_arr, 99)),
            max_ms=float(np.max(times_arr)),
            min_ms=float(np.min(times_arr)),
            std_ms=float(np.std(times_arr)),
            samples_per_sec=len(samples) / total_time if total_time > 0 else 0,
            total_samples=len(samples),
            passed=float(np.percentile(times_arr, 95)) < self.latency_threshold_ms,
            threshold_ms=self.latency_threshold_ms,
        )
    
    def test_batch_throughput(self,
                               predict_batch_fn: Callable[[List[str]], Any],
                               samples: List[str],
                               batch_sizes: List[int] = None) -> Dict[int, ThroughputResult]:
        """Test batch processing throughput."""
        if batch_sizes is None:
            batch_sizes = [32, 64, 128, 256]
        
        results = {}
        
        for batch_size in batch_sizes:
            # Create batches
            batches = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
            
            # Warmup
            if batches:
                predict_batch_fn(batches[0])
            
            # Measure
            start = time.perf_counter()
            for batch in batches:
                predict_batch_fn(batch)
            total_time = time.perf_counter() - start
            
            results[batch_size] = ThroughputResult(
                batch_size=batch_size,
                samples_per_sec=len(samples) / total_time if total_time > 0 else 0,
                batches_per_sec=len(batches) / total_time if total_time > 0 else 0,
                total_time_sec=total_time,
            )
        
        return results
    
    def test_memory_usage(self,
                          predict_fn: Callable[[str], Any],
                          samples: List[str]) -> MemoryResult:
        """Test memory usage during inference."""
        tracemalloc.start()
        
        for sample in samples:
            predict_fn(sample)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        
        return MemoryResult(
            current_mb=current_mb,
            peak_mb=peak_mb,
            passed=peak_mb < self.memory_threshold_mb,
            threshold_mb=self.memory_threshold_mb,
        )
    
    def test_long_input_latency(self,
                                 predict_fn: Callable[[str], Any],
                                 lengths: List[int] = None) -> Dict[int, float]:
        """Test latency vs input length."""
        if lengths is None:
            lengths = [100, 500, 1000, 5000, 10000]
        
        results = {}
        
        for length in lengths:
            sample = "A" * length
            
            # Warmup
            predict_fn(sample)
            
            # Measure (average of 5)
            times = []
            for _ in range(5):
                start = time.perf_counter()
                predict_fn(sample)
                times.append((time.perf_counter() - start) * 1000)
            
            results[length] = np.mean(times)
        
        return results
    
    def run_all(self,
                predict_fn: Callable[[str], Any],
                predict_batch_fn: Callable[[List[str]], Any],
                samples: List[str]) -> Dict[str, Any]:
        """Run all performance tests."""
        print("Running performance tests...")
        
        print("  Testing inference latency...")
        latency = self.test_inference_latency(predict_fn, samples[:500])
        
        print("  Testing batch throughput...")
        throughput = self.test_batch_throughput(predict_batch_fn, samples[:1000])
        
        print("  Testing memory usage...")
        memory = self.test_memory_usage(predict_fn, samples[:100])
        
        print("  Testing long input latency...")
        long_input = self.test_long_input_latency(predict_fn)
        
        return {
            'latency': latency,
            'throughput': throughput,
            'memory': memory,
            'long_input_latency': long_input,
            'all_passed': latency.passed and memory.passed,
        }
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format results as human-readable string."""
        lines = ["PERFORMANCE TEST RESULTS", "=" * 40]
        
        # Latency
        lat = results['latency']
        status = "✅" if lat.passed else "❌"
        lines.append(f"\nLatency {status}")
        lines.append(f"  Mean: {lat.mean_ms:.2f}ms")
        lines.append(f"  P50:  {lat.p50_ms:.2f}ms")
        lines.append(f"  P95:  {lat.p95_ms:.2f}ms (threshold: {lat.threshold_ms}ms)")
        lines.append(f"  P99:  {lat.p99_ms:.2f}ms")
        lines.append(f"  Max:  {lat.max_ms:.2f}ms")
        lines.append(f"  Throughput: {lat.samples_per_sec:.0f} samples/sec")
        
        # Throughput
        lines.append(f"\nBatch Throughput")
        for batch_size, tp in results['throughput'].items():
            lines.append(f"  Batch {batch_size}: {tp.samples_per_sec:.0f} samples/sec")
        
        # Memory
        mem = results['memory']
        status = "✅" if mem.passed else "❌"
        lines.append(f"\nMemory {status}")
        lines.append(f"  Current: {mem.current_mb:.1f}MB")
        lines.append(f"  Peak:    {mem.peak_mb:.1f}MB (threshold: {mem.threshold_mb}MB)")
        
        # Long input
        lines.append(f"\nLong Input Latency")
        for length, ms in results['long_input_latency'].items():
            lines.append(f"  {length} chars: {ms:.2f}ms")
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test with dummy model
    def dummy_predict(text: str) -> float:
        time.sleep(0.001)  # Simulate 1ms inference
        return 0.5
    
    def dummy_predict_batch(texts: List[str]) -> List[float]:
        time.sleep(0.001 * len(texts) * 0.5)  # Batch is faster
        return [0.5] * len(texts)
    
    suite = PerformanceSuite(latency_threshold_ms=50)
    samples = ["test sample " * 10] * 100
    
    results = suite.run_all(dummy_predict, dummy_predict_batch, samples)
    print(suite.format_results(results))
