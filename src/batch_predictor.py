"""Enhanced HybridPredictor with batch processing, validation, and monitoring."""
import torch
import joblib
import numpy as np
import logging
from urllib.parse import quote, urlsplit, urlunsplit
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Optional

from .input_validator import InputValidator, ValidationError
from .model_monitor import ModelMonitor


class BatchHybridPredictor:
    """Optimized predictor with batch processing, validation, and monitoring."""
    
    def __init__(self, models_dir='models', device=None, batch_size=256,
                 registry=None, validator: bool = True, monitor: ModelMonitor = None,
                 fail_on_no_models: bool = False):
        self.models_dir = Path(models_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.registry = registry
        self.validator = InputValidator() if validator else None
        self.monitor = monitor
        self.fail_on_no_models = fail_on_no_models
        self.sklearn_models = {}
        self.pytorch_models = {}
        self.load_errors = {}
        self.loaded = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
    
    def load_models(self):
        """Load all available models."""
        self.load_errors = {}
        # sklearn models
        for name, fname in [('network', 'network_intrusion_model.pkl'),
                           ('fraud', 'fraud_detection_model.pkl'),
                           ('url_lgbm', 'url_analysis_model.pkl')]:
            path = self.models_dir / fname
            if path.exists():
                try:
                    self.sklearn_models[name] = joblib.load(path)
                except Exception as exc:
                    self.load_errors[name] = str(exc)
                    self.logger.warning("Failed to load sklearn model %s from %s: %s", name, path, exc)
        
        # PyTorch models
        for name, fname in [('payload_cnn', 'payload_cnn.pt'),
                           ('url_cnn', 'url_cnn.pt'),
                           ('timeseries_lstm', 'timeseries_lstm.pt'),
                           ('meta_classifier', 'meta_classifier.pt')]:
            path = self.models_dir / fname
            if path.exists():
                try:
                    model = torch.jit.load(str(path), map_location=self.device)
                    model.eval()
                    self.pytorch_models[name] = model
                except Exception as exc:
                    self.load_errors[name] = str(exc)
                    self.logger.warning("Failed to load PyTorch model %s from %s: %s", name, path, exc)
        
        self.loaded = True
        if not self.sklearn_models and not self.pytorch_models:
            message = f"No models loaded from {self.models_dir}"
            self.logger.warning(message)
            if self.fail_on_no_models:
                raise FileNotFoundError(message)
        return self
    
    def _tokenize_batch(self, texts, max_len, vocab_size):
        """Batch tokenize texts to character indices."""
        batch = []
        for text in texts:
            chars = [ord(c) % vocab_size for c in str(text)[:max_len]]
            chars += [0] * (max_len - len(chars))
            batch.append(chars)
        return batch

    @staticmethod
    def _normalize_url_text(url: str) -> str:
        """Normalize URL to ASCII-safe text while preserving semantics for IDN/homograph cases."""
        text = str(url).strip()
        try:
            parts = urlsplit(text)
            netloc = parts.netloc.encode('idna').decode('ascii') if parts.netloc else parts.netloc
            path = quote(parts.path, safe="/:@-._~!$&'()*+,;=%")
            query = quote(parts.query, safe="=&?/:@-._~!$'()*+,;%")
            fragment = quote(parts.fragment, safe=":@-._~!$&'()*+,;=%")
            return urlunsplit((parts.scheme, netloc, path, query, fragment))
        except Exception:
            return text
    
    @torch.no_grad()
    def predict_payload_batch(self, payloads):
        """Batch predict for payloads with memory-efficient chunking."""
        if 'payload_cnn' not in self.pytorch_models:
            return np.full(len(payloads), 0.5)
        
        results = []
        for i in range(0, len(payloads), self.batch_size):
            chunk = payloads[i:i + self.batch_size]
            tokens = self._tokenize_batch(chunk, 500, 256)
            x = torch.tensor(tokens, dtype=torch.long, device=self.device)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                logits = self.pytorch_models['payload_cnn'](x)
            results.append(torch.sigmoid(logits).cpu().numpy())
        
        return np.concatenate(results).reshape(-1)
    
    @torch.no_grad()
    def predict_url_batch(self, urls):
        """Batch predict for URLs."""
        if 'url_cnn' not in self.pytorch_models:
            return np.full(len(urls), 0.5)
        
        results = []
        for i in range(0, len(urls), self.batch_size):
            chunk = urls[i:i + self.batch_size]
            normalized = [self._normalize_url_text(u) for u in chunk]
            tokens = self._tokenize_batch(normalized, 200, 128)
            x = torch.tensor(tokens, dtype=torch.long, device=self.device)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                logits = self.pytorch_models['url_cnn'](x)
            results.append(torch.sigmoid(logits).cpu().numpy())
        
        return np.concatenate(results).reshape(-1)
    
    @torch.no_grad()
    def predict_timeseries_batch(self, sequences):
        """Batch predict for time-series."""
        if 'timeseries_lstm' not in self.pytorch_models:
            return np.full(len(sequences), 0.5)
        
        sequences = np.array(sequences, dtype=np.float32)
        results = []
        
        for i in range(0, len(sequences), self.batch_size):
            chunk = sequences[i:i + self.batch_size]
            x = torch.tensor(chunk, dtype=torch.float32, device=self.device)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                logits = self.pytorch_models['timeseries_lstm'](x)
            results.append(torch.sigmoid(logits).cpu().numpy())
        
        return np.concatenate(results).reshape(-1)
    
    def predict_batch(self, data, validate: bool = True):
        """Full batch prediction pipeline with optional validation and monitoring."""
        if not self.loaded:
            self.load_models()

        if not data:
            raise ValidationError("No input data provided")

        working_data = dict(data)

        # Validate inputs if validator enabled
        if validate and self.validator:
            if 'payloads' in working_data:
                working_data['payloads'] = [self.validator.validate_payload(p) for p in working_data['payloads']]
            if 'urls' in working_data:
                working_data['urls'] = [self.validator.validate_url(u) for u in working_data['urls']]

        if 'timeseries' in working_data:
            ts = np.asarray(working_data['timeseries'], dtype=np.float32)
            if ts.ndim == 2:
                ts = ts[np.newaxis, ...]
            working_data['timeseries'] = ts

        modality_lengths = {}
        if 'payloads' in working_data:
            modality_lengths['payloads'] = len(working_data['payloads'])
        if 'urls' in working_data:
            modality_lengths['urls'] = len(working_data['urls'])
        if 'timeseries' in working_data:
            modality_lengths['timeseries'] = len(working_data['timeseries'])

        active_lengths = [v for v in modality_lengths.values() if v > 0]
        if not active_lengths:
            raise ValidationError("No non-empty input arrays provided")
        if len(set(active_lengths)) > 1:
            raise ValidationError(
                f"Mismatched batch sizes across modalities: {modality_lengths}. "
                "Send one modality per request or equal-length arrays."
            )
        n = active_lengths[0]
        
        scores = {}
        
        # Run predictions with optional monitoring
        if 'payloads' in working_data:
            if self.monitor:
                with self.monitor.track('payload_cnn') as ctx:
                    scores['payload'] = self.predict_payload_batch(working_data['payloads'])
                    ctx['confidence'] = scores['payload']
            else:
                scores['payload'] = self.predict_payload_batch(working_data['payloads'])
        
        if 'urls' in working_data:
            if self.monitor:
                with self.monitor.track('url_cnn') as ctx:
                    scores['url'] = self.predict_url_batch(working_data['urls'])
                    ctx['confidence'] = scores['url']
            else:
                scores['url'] = self.predict_url_batch(working_data['urls'])
        
        if 'timeseries' in working_data:
            if self.monitor:
                with self.monitor.track('timeseries_lstm') as ctx:
                    scores['timeseries'] = self.predict_timeseries_batch(working_data['timeseries'])
                    ctx['confidence'] = scores['timeseries']
            else:
                scores['timeseries'] = self.predict_timeseries_batch(working_data['timeseries'])

        for key, value in scores.items():
            if len(value) != n:
                raise RuntimeError(f"Model score length mismatch for {key}: got {len(value)}, expected {n}")
        
        # Fill missing with neutral
        for key in ['payload', 'url', 'timeseries', 'network', 'fraud']:
            if key not in scores:
                scores[key] = np.full(n, 0.5)
        
        # Ensemble
        model_scores = np.column_stack([scores['network'], scores['fraud'], 
                                        scores['url'], scores['payload'], scores['timeseries']])
        
        if 'meta_classifier' in self.pytorch_models:
            x = torch.tensor(model_scores, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                final = torch.sigmoid(self.pytorch_models['meta_classifier'](x)).cpu().numpy()
        else:
            final = np.average(model_scores, axis=1, weights=[0.3, 0.2, 0.2, 0.15, 0.15])
        
        return {'is_attack': (final > 0.5).astype(int), 'confidence': final, 'scores': scores}
    
    def predict_async(self, data, callback=None):
        """Async prediction with optional callback."""
        future = self._executor.submit(self.predict_batch, data)
        if callback:
            future.add_done_callback(lambda f: callback(f.result()))
        return future
    
    def benchmark(self, n_samples=1000):
        """Benchmark throughput."""
        payloads = ['test payload ' * 10] * n_samples
        
        start = time.time()
        self.predict_payload_batch(payloads)
        elapsed = time.time() - start
        
        throughput = n_samples / elapsed
        print(f"Throughput: {throughput:.0f} samples/sec ({elapsed:.2f}s for {n_samples})")
        return throughput


def create_batch_predictor(models_dir='models', batch_size=256, 
                           enable_validation=True, enable_monitoring=False,
                           fail_on_no_models: bool = False):
    """Factory function with optional validation and monitoring."""
    monitor = ModelMonitor() if enable_monitoring else None
    return BatchHybridPredictor(
        models_dir, batch_size=batch_size, 
        validator=enable_validation, monitor=monitor,
        fail_on_no_models=fail_on_no_models
    ).load_models()
