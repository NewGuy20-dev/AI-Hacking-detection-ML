"""Difficulty-based obfuscation for stress test scenarios."""
import random
import urllib.parse
import numpy as np
from typing import Union, Any


class DifficultyMixin:
    """Applies obfuscation based on difficulty tier."""
    
    @staticmethod
    def apply_difficulty(data: Any, difficulty: str, data_type: str) -> Any:
        """
        Apply obfuscation based on difficulty tier.
        
        Args:
            data: Input data (str for payload/url, np.ndarray for timeseries/tabular)
            difficulty: 'easy', 'medium', 'hard', or 'adversarial'
            data_type: 'payload', 'url', 'timeseries', or 'tabular'
        
        Returns:
            Obfuscated data in same format as input
        """
        if difficulty == 'easy':
            return data  # No obfuscation
        
        if data_type == 'payload':
            return DifficultyMixin._obfuscate_payload(data, difficulty)
        elif data_type == 'url':
            return DifficultyMixin._obfuscate_url(data, difficulty)
        elif data_type == 'timeseries':
            return DifficultyMixin._obfuscate_timeseries(data, difficulty)
        elif data_type == 'tabular':
            return DifficultyMixin._obfuscate_tabular(data, difficulty)
        else:
            return data
    
    @staticmethod
    def _obfuscate_payload(payload: str, difficulty: str) -> str:
        """Apply payload obfuscation techniques."""
        if difficulty == 'medium':
            # Single encoding
            techniques = [
                lambda p: urllib.parse.quote(p),  # URL encode
                lambda p: p.swapcase(),  # Case swap
                lambda p: p.replace(' ', '/**/'),  # Comment injection
            ]
            return random.choice(techniques)(payload)
        
        elif difficulty == 'hard':
            # Double encoding + case mixing
            p = payload
            if random.random() < 0.5:
                p = urllib.parse.quote(urllib.parse.quote(p))  # Double encode
            if random.random() < 0.5:
                p = ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in p)  # Case mix
            if random.random() < 0.3:
                p = p.replace(' ', random.choice(['/**/', '\t', '\n', '+']))  # Whitespace variation
            return p
        
        elif difficulty == 'adversarial':
            # Triple encoding + polyglots + null bytes
            p = payload
            if random.random() < 0.4:
                # Triple encode
                p = urllib.parse.quote(urllib.parse.quote(urllib.parse.quote(p)))
            if random.random() < 0.3:
                # Null byte injection
                parts = list(p)
                if len(parts) > 2:
                    parts.insert(random.randint(1, len(parts)-1), '\x00')
                p = ''.join(parts)
            if random.random() < 0.3:
                # Polyglot (combine SQL + XSS)
                p = f"'\"><script>{p}</script>"
            if random.random() < 0.3:
                # Fragmentation
                p = p.replace('OR', 'O/**/R').replace('AND', 'A/**/ND')
            return p
        
        return payload
    
    @staticmethod
    def _obfuscate_url(url: str, difficulty: str) -> str:
        """Apply URL obfuscation techniques."""
        # Homograph mappings
        CYRILLIC = {'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'x': 'х'}
        GREEK = {'o': 'ο', 'a': 'α', 'v': 'ν', 'i': 'ι'}
        
        if difficulty == 'medium':
            # Single typo or simple obfuscation
            techniques = [
                lambda u: u.replace('a', '4', 1),  # Leet speak
                lambda u: u.replace('o', '0', 1),
                lambda u: u.replace('i', '1', 1),
                lambda u: u.replace('.com', '.co'),  # TLD typo
            ]
            return random.choice(techniques)(url)
        
        elif difficulty == 'hard':
            # Homograph + encoding
            u = url
            if random.random() < 0.5 and any(c in u for c in CYRILLIC.keys()):
                # Cyrillic substitution
                for latin, cyrillic in CYRILLIC.items():
                    if latin in u and random.random() < 0.3:
                        u = u.replace(latin, cyrillic, 1)
            if random.random() < 0.3:
                # Add zero-width space
                parts = list(u)
                if len(parts) > 5:
                    parts.insert(random.randint(3, len(parts)-2), '\u200B')
                u = ''.join(parts)
            return u
        
        elif difficulty == 'adversarial':
            # Advanced obfuscation
            u = url
            if random.random() < 0.3:
                # Greek homograph
                for latin, greek in GREEK.items():
                    if latin in u and random.random() < 0.4:
                        u = u.replace(latin, greek, 1)
            if random.random() < 0.3:
                # IP obfuscation (convert domain to decimal IP)
                if 'http://' in u:
                    u = f"http://{random.randint(3000000000, 4000000000)}/"
            if random.random() < 0.2:
                # Data URI
                u = f"data:text/html,<script>location='{u}'</script>"
            if random.random() < 0.3:
                # Punycode-like obfuscation
                u = u.replace('.com', '.xn--com')
            return u
        
        return url
    
    @staticmethod
    def _obfuscate_timeseries(data: np.ndarray, difficulty: str) -> np.ndarray:
        """Apply timeseries obfuscation (gradual attacks, noise)."""
        # data shape: (60, 8)
        if difficulty == 'medium':
            # Add moderate noise
            noise = np.random.normal(0, 0.1, data.shape)
            return data + noise
        
        elif difficulty == 'hard':
            # Add significant noise
            noise = np.random.normal(0, 0.3, data.shape)
            return data + noise
        
        elif difficulty == 'adversarial':
            # Heavy noise + subtle drift
            noise = np.random.normal(0, 0.5, data.shape)
            drift = np.linspace(0, 0.2, data.shape[0])[:, np.newaxis]
            return data + noise + drift
        
        return data
    
    @staticmethod
    def _obfuscate_tabular(data: np.ndarray, difficulty: str) -> np.ndarray:
        """Apply tabular data obfuscation (noise, outliers)."""
        if difficulty == 'medium':
            # Small noise
            noise = np.random.normal(0, 0.05, data.shape)
            return data + noise
        
        elif difficulty == 'hard':
            # Moderate noise + occasional outliers
            noise = np.random.normal(0, 0.15, data.shape)
            result = data + noise
            # Add outliers to 10% of features
            outlier_mask = np.random.random(data.shape) < 0.1
            result[outlier_mask] *= random.uniform(1.5, 3.0)
            return result
        
        elif difficulty == 'adversarial':
            # Heavy noise + frequent outliers
            noise = np.random.normal(0, 0.3, data.shape)
            result = data + noise
            # Add outliers to 20% of features
            outlier_mask = np.random.random(data.shape) < 0.2
            result[outlier_mask] *= random.uniform(2.0, 5.0)
            return result
        
        return data
