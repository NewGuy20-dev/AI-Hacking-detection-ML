"""Input validation and sanitization for ML predictions."""
import re
from typing import List, Tuple, Union
import numpy as np


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


class InputValidator:
    """Validates and sanitizes inputs before model prediction."""
    
    MAX_PAYLOAD_LEN = 10000
    MAX_URL_LEN = 2000
    TRUNCATE_PAYLOAD = 500
    TRUNCATE_URL = 200
    EXPECTED_NETWORK_FEATURES = 41
    
    @staticmethod
    def validate_payload(payload: str, truncate: bool = True) -> str:
        """Validate and sanitize a payload string."""
        if not isinstance(payload, str):
            raise ValidationError(f"Payload must be string, got {type(payload).__name__}")
        if len(payload) > InputValidator.MAX_PAYLOAD_LEN:
            raise ValidationError(f"Payload too long: {len(payload)} > {InputValidator.MAX_PAYLOAD_LEN}")
        payload = payload.replace('\x00', '')  # Remove null bytes
        if truncate:
            return payload[:InputValidator.TRUNCATE_PAYLOAD]
        return payload
    
    @staticmethod
    def validate_url(url: str, truncate: bool = True) -> str:
        """Validate and sanitize a URL string."""
        if not isinstance(url, str):
            raise ValidationError(f"URL must be string, got {type(url).__name__}")
        if len(url) > InputValidator.MAX_URL_LEN:
            raise ValidationError(f"URL too long: {len(url)} > {InputValidator.MAX_URL_LEN}")
        url = url.strip()
        if not re.match(r'^https?://', url, re.IGNORECASE):
            url = f"http://{url}"
        if truncate:
            return url[:InputValidator.TRUNCATE_URL]
        return url
    
    @staticmethod
    def validate_network_features(features: Union[List, np.ndarray], 
                                   expected_dim: int = None) -> np.ndarray:
        """Validate network feature array."""
        expected_dim = expected_dim or InputValidator.EXPECTED_NETWORK_FEATURES
        try:
            arr = np.asarray(features, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Features must be numeric array: {e}")
        
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != expected_dim:
            raise ValidationError(f"Expected {expected_dim} features, got {arr.shape[-1]}")
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValidationError("Features contain NaN or Inf values")
        return arr
    
    @staticmethod
    def validate_batch(items: List, validator_fn, collect_errors: bool = False
                       ) -> Union[List, Tuple[List, List]]:
        """Validate a batch of items, optionally collecting errors."""
        valid, errors = [], []
        for i, item in enumerate(items):
            try:
                valid.append(validator_fn(item))
            except ValidationError as e:
                if collect_errors:
                    errors.append((i, str(e)))
                else:
                    raise ValidationError(f"Item {i}: {e}")
        if collect_errors:
            return valid, errors
        return valid
