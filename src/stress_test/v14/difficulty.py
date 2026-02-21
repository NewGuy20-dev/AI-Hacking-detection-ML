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
        parsed = urllib.parse.urlsplit(url if isinstance(url, str) else str(url))
        scheme = parsed.scheme or 'http'
        netloc = parsed.netloc
        path = parsed.path or '/'
        query = parsed.query
        fragment = parsed.fragment

        # Normalize bare host inputs
        if not netloc and parsed.path and '://' not in url:
            netloc = parsed.path
            path = '/'
        
        def rebuild(nl: str, p: str, q: str, f: str) -> str:
            safe_path = urllib.parse.quote(p, safe="/:@-._~!$&'()*+,;=%")
            return urllib.parse.urlunsplit((scheme, nl, safe_path, q, f))
        
        if difficulty == 'medium':
            # Single typo or simple obfuscation
            techniques = [
                lambda nl: nl.replace('a', '4', 1),  # Leet speak
                lambda nl: nl.replace('o', '0', 1),
                lambda nl: nl.replace('i', '1', 1),
                lambda nl: nl.replace('.com', '.co'),  # TLD typo
            ]
            new_netloc = random.choice(techniques)(netloc)
            return rebuild(new_netloc, path, query, fragment)
        
        elif difficulty == 'hard':
            # Homograph + encoding
            new_netloc = netloc
            new_path = path
            if random.random() < 0.5 and any(c in new_netloc for c in CYRILLIC.keys()):
                # Cyrillic substitution
                for latin, cyrillic in CYRILLIC.items():
                    if latin in new_netloc and random.random() < 0.3:
                        new_netloc = new_netloc.replace(latin, cyrillic, 1)
            if random.random() < 0.3:
                # Encode path fragments while keeping URL parse-valid.
                segments = [urllib.parse.quote(seg, safe='') for seg in new_path.split('/')]
                new_path = '/'.join(segments)
            return rebuild(new_netloc, new_path, query, fragment)
        
        elif difficulty == 'adversarial':
            # Advanced obfuscation
            new_netloc = netloc
            new_path = path
            new_query = query
            if random.random() < 0.3:
                # Greek homograph
                for latin, greek in GREEK.items():
                    if latin in new_netloc and random.random() < 0.4:
                        new_netloc = new_netloc.replace(latin, greek, 1)
            if random.random() < 0.3:
                # IP-like host replacement while preserving parse structure.
                new_netloc = '.'.join(str(random.randint(1, 254)) for _ in range(4))
            if random.random() < 0.2:
                target = urllib.parse.quote(rebuild(new_netloc, new_path, new_query, fragment), safe='')
                new_query = f"redirect={target}"
            if random.random() < 0.3:
                # Punycode-like visual marker
                new_netloc = new_netloc.replace('.com', '.xn--com')
            return rebuild(new_netloc, new_path, new_query, fragment)
        
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
