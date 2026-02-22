"""Shadow logging for opt-in live evaluation."""
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class ShadowLogger:
    def __init__(self, log_path: Path, store_raw: bool = False):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.store_raw = store_raw

    @staticmethod
    def _hash_input(payload: str) -> str:
        return hashlib.sha256(payload.encode('utf-8', errors='ignore')).hexdigest()

    def log(self, *, model: str, route: str, input_data: str, prediction: int,
            confidence: float, latency_ms: float, version: str = "", error: Optional[str] = None):
        input_hash = self._hash_input(input_data) if input_data is not None else None
        record = {
            'ts': datetime.utcnow().isoformat() + 'Z',
            'model': model,
            'route': route,
            'input_hash': input_hash,
            'input_len': len(input_data) if input_data is not None else 0,
            'input_type': 'text',
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'version': version,
            'error': error,
        }
        if self.store_raw:
            record['raw_input'] = input_data[:500]
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

