"""Training monitor for GPU utilization and batch timing diagnostics."""
import time
import logging
import subprocess
import threading
from typing import Optional, Callable
from collections import deque

import torch

from .config import RobustTrainingConfig


class GPUMonitor:
    """Monitor GPU utilization with pynvml or nvidia-smi fallback."""
    
    def __init__(self):
        self.logger = logging.getLogger('GPUMonitor')
        self._pynvml_available = False
        self._handle = None
        self._init_pynvml()
    
    def _init_pynvml(self) -> None:
        """Try to initialize pynvml."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml_available = True
            self.logger.debug("Using pynvml for GPU monitoring")
        except Exception:
            self.logger.debug("pynvml not available, using nvidia-smi fallback")
    
    def get_utilization(self) -> Optional[float]:
        """Get GPU utilization percentage (0-100)."""
        if self._pynvml_available:
            return self._get_util_pynvml()
        return self._get_util_nvidia_smi()
    
    def _get_util_pynvml(self) -> Optional[float]:
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return float(util.gpu)
        except Exception:
            return None
    
    def _get_util_nvidia_smi(self) -> Optional[float]:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split('\n')[0])
        except Exception:
            pass
        return None
    
    def get_memory_info(self) -> Optional[dict]:
        """Get GPU memory usage."""
        if self._pynvml_available:
            try:
                import pynvml
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return {
                    'used_mb': info.used / 1024 / 1024,
                    'total_mb': info.total / 1024 / 1024,
                    'percent': 100 * info.used / info.total,
                }
            except Exception:
                pass
        return None


class TrainingMonitor:
    """Background monitor for training health diagnostics.
    
    Features:
    - GPU utilization monitoring with idle detection
    - Batch timing with anomaly detection
    - Memory usage tracking
    - Configurable alert callbacks
    """
    
    def __init__(
        self,
        config: Optional[RobustTrainingConfig] = None,
        on_gpu_idle: Optional[Callable[[float], None]] = None,
        on_slow_batch: Optional[Callable[[int, float], None]] = None,
    ):
        self.config = config or RobustTrainingConfig()
        self.on_gpu_idle = on_gpu_idle
        self.on_slow_batch = on_slow_batch
        
        self.logger = logging.getLogger('TrainingMonitor')
        self.gpu_monitor = GPUMonitor() if torch.cuda.is_available() else None
        
        # State tracking
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._training_active = False
        self._last_batch_time = time.time()
        self._idle_start: Optional[float] = None
        
        # Batch timing
        self._batch_times = deque(maxlen=50)
    
    def start(self) -> None:
        """Start background monitoring thread."""
        if self._running or not self.config.enable_monitoring:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info("Training monitor started")
    
    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        self.logger.info("Training monitor stopped")
    
    def set_training_active(self, active: bool) -> None:
        """Mark whether training is actively processing batches."""
        self._training_active = active
        if active:
            self._idle_start = None
    
    def record_batch_time(self, batch_idx: int, duration: float) -> None:
        """Record batch processing time and check for anomalies."""
        self._last_batch_time = time.time()
        self._batch_times.append(duration)
        
        # Check for slow batch
        if len(self._batch_times) >= 10:
            avg = sum(self._batch_times) / len(self._batch_times)
            if duration > self.config.batch_time_warning_multiplier * avg:
                self.logger.warning(
                    f"Slow batch {batch_idx}: {duration:.2f}s (avg: {avg:.2f}s)"
                )
                if self.on_slow_batch:
                    self.on_slow_batch(batch_idx, duration)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                self._check_gpu_utilization()
            except Exception as e:
                self.logger.debug(f"Monitor error: {e}")
            
            time.sleep(self.config.gpu_check_interval)
    
    def _check_gpu_utilization(self) -> None:
        """Check GPU utilization and detect idle periods."""
        if not self.gpu_monitor or not self._training_active:
            return
        
        util = self.gpu_monitor.get_utilization()
        if util is None:
            return
        
        if util < self.config.gpu_idle_threshold:
            if self._idle_start is None:
                self._idle_start = time.time()
            elif time.time() - self._idle_start > self.config.gpu_idle_duration:
                self.logger.warning(
                    f"GPU idle ({util:.1f}%) for {time.time() - self._idle_start:.0f}s "
                    f"while training active - possible data loading stall"
                )
                if self.on_gpu_idle:
                    self.on_gpu_idle(util)
                self._idle_start = time.time()  # Reset to avoid spam
        else:
            self._idle_start = None
    
    def get_stats(self) -> dict:
        """Get current monitoring statistics."""
        stats = {
            'avg_batch_time': sum(self._batch_times) / len(self._batch_times) if self._batch_times else 0,
            'batch_count': len(self._batch_times),
        }
        
        if self.gpu_monitor:
            stats['gpu_util'] = self.gpu_monitor.get_utilization()
            mem = self.gpu_monitor.get_memory_info()
            if mem:
                stats['gpu_memory_percent'] = mem['percent']
        
        return stats
    
    def __enter__(self) -> 'TrainingMonitor':
        self.start()
        return self
    
    def __exit__(self, *args) -> None:
        self.stop()
