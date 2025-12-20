"""Phase D1: Enhanced HybridPredictor with batch processing optimization."""
import torch
import joblib
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time


class BatchHybridPredictor:
    """Optimized predictor with batch processing and async support."""
    
    def __init__(self, models_dir='models', device=None, batch_size=256):
        self.models_dir = Path(models_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.sklearn_models = {}
        self.pytorch_models = {}
        self.loaded = False
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def load_models(self):
        """Load all available models."""
        # sklearn models
        for name, fname in [('network', 'network_intrusion_model.pkl'),
                           ('fraud', 'fraud_detection_model.pkl'),
                           ('url_lgbm', 'url_analysis_model.pkl')]:
            path = self.models_dir / fname
            if path.exists():
                try:
                    self.sklearn_models[name] = joblib.load(path)
                except: pass
        
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
                except: pass
        
        self.loaded = True
        return self
    
    def _tokenize_batch(self, texts, max_len, vocab_size):
        """Batch tokenize texts to character indices."""
        batch = []
        for text in texts:
            chars = [ord(c) % vocab_size for c in str(text)[:max_len]]
            chars += [0] * (max_len - len(chars))
            batch.append(chars)
        return batch
    
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
        
        return np.concatenate(results)
    
    @torch.no_grad()
    def predict_url_batch(self, urls):
        """Batch predict for URLs."""
        if 'url_cnn' not in self.pytorch_models:
            return np.full(len(urls), 0.5)
        
        results = []
        for i in range(0, len(urls), self.batch_size):
            chunk = urls[i:i + self.batch_size]
            tokens = self._tokenize_batch(chunk, 200, 128)
            x = torch.tensor(tokens, dtype=torch.long, device=self.device)
            
            with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                logits = self.pytorch_models['url_cnn'](x)
            results.append(torch.sigmoid(logits).cpu().numpy())
        
        return np.concatenate(results)
    
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
        
        return np.concatenate(results)
    
    def predict_batch(self, data):
        """Full batch prediction pipeline."""
        if not self.loaded:
            self.load_models()
        
        n = max(len(data.get('payloads', [])), len(data.get('urls', [])), 
                len(data.get('timeseries', [[]])), 1)
        
        scores = {}
        
        # Run predictions
        if 'payloads' in data:
            scores['payload'] = self.predict_payload_batch(data['payloads'])
        if 'urls' in data:
            scores['url'] = self.predict_url_batch(data['urls'])
        if 'timeseries' in data:
            scores['timeseries'] = self.predict_timeseries_batch(data['timeseries'])
        
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


def create_batch_predictor(models_dir='models', batch_size=256):
    """Factory function."""
    return BatchHybridPredictor(models_dir, batch_size=batch_size).load_models()
