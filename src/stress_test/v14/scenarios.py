"""Scenario dataclasses and registry for V1.4 stress test suite."""
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Tuple
from pathlib import Path
import yaml
import random
import urllib.parse
import numpy as np
import json


@dataclass
class Scenario:
    """Test scenario definition."""
    id: str
    model: str
    category: str
    subcategory: str
    input_data: Any
    expected_label: int
    difficulty: str
    description: str
    source: str
    tags: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result from running a scenario."""
    scenario: Scenario
    prediction: int
    confidence: float
    passed: bool
    latency_ms: float
    timestamp: str
    error: Optional[str] = None


class ScenarioRegistry:
    """Registry for loading and managing scenarios."""
    
    def __init__(self, scenarios_dir: Path):
        self.scenarios_dir = Path(scenarios_dir)
        
    def load_static(self, model_name: str) -> List[Scenario]:
        """Load static scenarios from YAML for a model."""
        yaml_path = self.scenarios_dir / f"{model_name}.yaml"
        
        if not yaml_path.exists():
            return []
        
        with open(yaml_path, encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        scenarios = []
        for item in data.get('scenarios', []):
            input_data = item.get('input')
            input_file = item.get('input_file')
            if input_file:
                input_data = self._load_input_file(model_name, str(input_file))
            elif input_data == 'DYNAMIC':
                # Skip generator placeholders when no explicit static fixture exists.
                continue
            input_data = self._coerce_static_input(model_name, input_data)
                
            scenarios.append(Scenario(
                id=item['id'],
                model=model_name,
                category=item['category'],
                subcategory=item['subcategory'],
                input_data=input_data,
                expected_label=item['expected'],
                difficulty=item['difficulty'],
                description=item['description'],
                source='static',
                tags=item.get('tags', []),
            ))
        
        return scenarios

    def _load_input_file(self, model_name: str, input_file: str) -> Any:
        path = Path(input_file)
        if not path.is_absolute():
            path = self.scenarios_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Static fixture file not found for {model_name}: {path}")
        suffix = path.suffix.lower()
        if suffix == '.json':
            with open(path, 'r', encoding='utf-8') as handle:
                return json.load(handle)
        if suffix in {'.txt', '.payload', '.url'}:
            return path.read_text(encoding='utf-8').strip()
        if suffix == '.npy':
            return np.load(path, allow_pickle=False)
        return path.read_text(encoding='utf-8').strip()

    @staticmethod
    def _coerce_static_input(model_name: str, input_data: Any) -> Any:
        if isinstance(input_data, dict) and 'input' in input_data:
            input_data = input_data['input']

        if model_name == 'timeseries':
            if isinstance(input_data, np.ndarray):
                return input_data.astype(np.float32, copy=False)
            if isinstance(input_data, list):
                return np.asarray(input_data, dtype=np.float32)
            return input_data

        if model_name in {'fraud', 'host', 'network', 'anomaly', 'meta'} and isinstance(input_data, np.ndarray):
            return input_data.astype(np.float32, copy=False).tolist()
        return input_data


# Dynamic Generators

class DynamicGenerator:
    """Base class for dynamic scenario generation."""
    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    @staticmethod
    def _normalize_weights(
        category_weights: Dict[str, float],
        fallback_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Return a normalized malicious-only weight map."""
        source = category_weights or fallback_weights
        filtered = {k: float(v) for k, v in source.items() if k in fallback_weights and v > 0}
        if not filtered:
            filtered = dict(fallback_weights)
        total = sum(filtered.values()) or 1.0
        return {k: v / total for k, v in filtered.items()}
    
    def generate(self, count: int, category_weights: Dict[str, float]) -> List[Scenario]:
        raise NotImplementedError


class PayloadGenerator(DynamicGenerator):
    """Generate payload variations using mutation techniques with real data and difficulty tiers."""

    DEFAULT_MALICIOUS_WEIGHTS = {
        'sqli': 0.25, 'xss': 0.20, 'cmdi': 0.20, 'path_traversal': 0.15,
        'ssti': 0.10, 'xxe': 0.05, 'ldap': 0.05
    }
    
    BASE_PAYLOADS = {
        'sqli': ["' OR '1'='1", "' UNION SELECT NULL--", "'; DROP TABLE users--", "' AND 1=1--"],
        'xss': ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "<svg onload=alert(1)>"],
        'cmdi': ["| cat /etc/passwd", "; ls -la", "$(whoami)", "`id`"],
        'path_traversal': ["../../../etc/passwd", "..\\..\\..\\windows\\system32\\config\\sam"],
        'ssti': ["{{7*7}}", "${7*7}", "{{config.items()}}"],
        'xxe': ["<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>"],
        'ldap': ["*)(uid=*))(|(uid=*", "admin)(&)"],
    }
    
    BENIGN_PAYLOADS = [
        # Normal text (~23%)
        "Order #12345 has been shipped.",
        "Hello, my name is John Smith.",
        "Please contact us at support@example.com.",
        "Meeting scheduled for 2026-12-15.",
        "The weather today is sunny.",
        # Code snippets (~77%)
        "function calculateTotal(items) { return items.reduce((sum, item) => sum + item.price, 0); }",
        "for i in range(10): print(i)",
        "const greeting = 'Hello World';",
        "class User { constructor(name) { this.name = name; } }",
        "def process_data(data): return data.strip()",
        "function add(a, b) { return a + b; }",
        "for (let i = 0; i < 10; i++) { console.log(i); }",
        "import React from 'react';",
        "class Product:\n    def __init__(self):\n        self.price = 99",
        "const items = [1, 2, 3, 4, 5];",
    ]
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        from .real_data import RealDataLoader
        from .difficulty import DifficultyMixin
        self.real_loader = RealDataLoader()
        self.difficulty_mixin = DifficultyMixin()
    
    def generate(self, count: int, category_weights: Dict[str, float], benign_ratio: float = 0.7) -> List[Scenario]:
        scenarios = []
        benign_count = int(count * benign_ratio)
        malicious_count = count - benign_count
        
        # Generate benign samples
        scenarios.extend(self._generate_benign(benign_count))
        
        # Generate malicious samples
        scenarios.extend(self._generate_malicious(malicious_count, category_weights))
        
        return scenarios
    
    def _generate_benign(self, count: int) -> List[Scenario]:
        scenarios = []
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        for i in range(count):
            payload = random.choice(self.BENIGN_PAYLOADS)
            difficulty = random.choice(difficulties)
            scenarios.append(Scenario(
                id=f"payload_benign_{i}_{random.randint(1000,9999)}",
                model='payload',
                category='benign',
                subcategory='normal',
                input_data=payload,
                expected_label=0,
                difficulty=difficulty,
                description='Benign payload',
                source='dynamic'
            ))
        return scenarios
    
    def _generate_malicious(self, count: int, category_weights: Dict[str, float]) -> List[Scenario]:
        scenarios = []
        normalized_weights = self._normalize_weights(category_weights, self.DEFAULT_MALICIOUS_WEIGHTS)
        categories = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        
        for i in range(count):
            category = random.choices(categories, weights=weights)[0]
            difficulty = random.choice(difficulties)
            
            # 50% real data, 50% synthetic
            if random.random() < 0.5 and category in ['sqli', 'xss', 'cmdi', 'path_traversal', 'ssti', 'xxe', 'ldap']:
                real_category_map = {
                    'sqli': 'sqli',
                    'xss': 'xss',
                    'cmdi': 'cmdi',
                    'path_traversal': 'path',
                    'ssti': 'ssti',
                    'xxe': 'xxe',
                    'ldap': 'ldap'
                }
                real_samples = self.real_loader.sample(real_category_map.get(category, 'sqli'), 1)
                payload = real_samples[0] if real_samples else random.choice(self.BASE_PAYLOADS.get(category, ["test"]))
            else:
                payload = random.choice(self.BASE_PAYLOADS.get(category, ["test"]))
            
            obfuscated = self.difficulty_mixin.apply_difficulty(payload, difficulty, 'payload')
            
            scenarios.append(Scenario(
                id=f"payload_mal_{i}_{random.randint(1000,9999)}",
                model='payload',
                category=category,
                subcategory='dynamic',
                input_data=obfuscated,
                expected_label=1,
                difficulty=difficulty,
                description=f'Dynamic {category} variant ({difficulty})',
                source='dynamic'
            ))
        return scenarios


class URLGenerator(DynamicGenerator):
    """Generate URL variations with real data and difficulty tiers."""
    
    DEFAULT_MALICIOUS_WEIGHTS = {
        'phishing': 0.30,
        'typosquatting': 0.25,
        'shorteners': 0.15,
        'homograph': 0.15,
        'dga': 0.10,
        'malware': 0.05
    }
    BRANDS = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'netflix']
    TLDS = ['.com', '.net', '.org', '.co', '.io', '.info']
    
    BENIGN_URLS = [
        "https://www.google.com/search?q=python+tutorial",
        "https://github.com/torvalds/linux",
        "https://stackoverflow.com/questions/tagged/javascript",
        "https://www.amazon.com/books/bestsellers",
        "https://docs.python.org/3/library/",
        "https://www.wikipedia.org/wiki/Machine_learning",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.reddit.com/r/programming",
        "https://news.ycombinator.com/",
        "https://www.microsoft.com/en-us/windows",
        "https://www.apple.com/iphone",
        "https://www.netflix.com/browse",
        "https://twitter.com/home",
        "https://www.linkedin.com/feed/",
        "https://medium.com/@author/article-title",
    ]
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        from .real_data import RealDataLoader
        from .difficulty import DifficultyMixin
        self.real_loader = RealDataLoader()
        self.difficulty_mixin = DifficultyMixin()
    
    def generate(self, count: int, category_weights: Dict[str, float], benign_ratio: float = 0.7) -> List[Scenario]:
        scenarios = []
        benign_count = int(count * benign_ratio)
        malicious_count = count - benign_count
        
        # Generate benign samples
        scenarios.extend(self._generate_benign(benign_count))
        
        # Generate malicious samples
        scenarios.extend(self._generate_malicious(malicious_count, category_weights))
        
        return scenarios
    
    def _generate_benign(self, count: int) -> List[Scenario]:
        scenarios = []
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        for i in range(count):
            url = random.choice(self.BENIGN_URLS)
            difficulty = random.choice(difficulties)
            scenarios.append(Scenario(
                id=f"url_benign_{i}_{random.randint(1000,9999)}",
                model='url',
                category='benign',
                subcategory='normal',
                input_data=url,
                expected_label=0,
                difficulty=difficulty,
                description='Benign URL',
                source='dynamic'
            ))
        return scenarios
    
    def _generate_malicious(self, count: int, category_weights: Dict[str, float]) -> List[Scenario]:
        scenarios = []
        normalized_weights = self._normalize_weights(category_weights, self.DEFAULT_MALICIOUS_WEIGHTS)
        categories = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        
        for i in range(count):
            category = random.choices(categories, weights=weights)[0]
            difficulty = random.choice(difficulties)
            
            # 50% real data, 50% synthetic
            if random.random() < 0.5 and category in ['phishing', 'malware']:
                real_category = 'phishing_url' if category == 'phishing' else 'malware_url'
                real_samples = self.real_loader.sample(real_category, 1)
                url = real_samples[0] if real_samples else self._generate_generic(category)
            else:
                if category == 'phishing':
                    url = self._generate_phishing()
                elif category == 'typosquatting':
                    url = self._generate_typosquatting()
                elif category == 'shorteners':
                    url = self._generate_shortener()
                elif category == 'homograph':
                    url = self._generate_homograph()
                elif category == 'dga':
                    url = self._generate_dga()
                elif category == 'malware':
                    url = self._generate_malware()
                else:
                    url = self._generate_generic(category)
            
            obfuscated = self.difficulty_mixin.apply_difficulty(url, difficulty, 'url')
            
            scenarios.append(Scenario(
                id=f"url_mal_{i}_{random.randint(1000,9999)}",
                model='url',
                category=category,
                subcategory='dynamic',
                input_data=obfuscated,
                expected_label=1,
                difficulty=difficulty,
                description=f'Dynamic {category} URL ({difficulty})',
                source='dynamic'
            ))
        return scenarios
    
    def _generate_phishing(self) -> str:
        brand = random.choice(self.BRANDS)
        keywords = ['verify', 'secure', 'account', 'login', 'update', 'confirm']
        host = f"{brand}-{random.choice(keywords)}{random.choice(self.TLDS)}"
        return f"http://{self._host_with_optional_port(host)}{self._random_path()}{self._random_query()}"
    
    def _generate_typosquatting(self) -> str:
        brand = random.choice(self.BRANDS)
        typo = brand[:-1] + random.choice('abcdefghijklmnopqrstuvwxyz')
        host = f"{typo}{random.choice(self.TLDS)}"
        return f"http://{self._host_with_optional_port(host)}{self._random_path()}"

    def _generate_shortener(self) -> str:
        short_domains = ['bit.ly', 'tinyurl.com', 'is.gd', 't.co', 'cutt.ly']
        token = ''.join(random.choices('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=7))
        if random.random() < 0.5:
            target = urllib.parse.quote(self._generate_generic('redirect'))
            return f"http://{random.choice(short_domains)}/{token}?url={target}"
        return f"http://{random.choice(short_domains)}/{token}"

    def _generate_homograph(self) -> str:
        # Mix Latin and confusable characters to mimic realistic homograph abuse.
        template = random.choice(['google', 'paypal', 'microsoft', 'apple'])
        mappings = {'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'x': 'х', 'i': 'і'}
        chars = []
        for ch in template:
            if ch in mappings and random.random() < 0.5:
                chars.append(mappings[ch])
            else:
                chars.append(ch)
        domain = ''.join(chars)
        host = f"{domain}{random.choice(self.TLDS)}"
        return f"http://{self._host_with_optional_port(host)}{self._random_path()}"
    
    def _generate_dga(self) -> str:
        length = random.randint(8, 16)
        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
        domain = ''.join(random.choices(alphabet, k=length))
        host = f"{domain}{random.choice(self.TLDS)}"
        return f"http://{self._host_with_optional_port(host)}{self._random_path()}"

    def _generate_malware(self) -> str:
        filename = random.choice(['update.exe', 'invoice.zip', 'scan.js', 'loader.bin', 'payload.dll'])
        lure_path = random.choice(['/download', '/secure', '/patch', '/driver', '/cdn'])
        host = f"cdn-{random.randint(100,999)}.{random.choice(['ru', 'tk', 'top', 'xyz'])}"
        return f"http://{self._host_with_optional_port(host)}{lure_path}/{filename}{self._random_query()}"
    
    def _generate_generic(self, category: str) -> str:
        if random.random() < 0.2:
            # Decimal-ish IPv4 literals are common in malicious URL feeds.
            host = ".".join(str(random.randint(1, 254)) for _ in range(4))
        else:
            host = f"malicious-{category}-{random.randint(1000, 9999)}.com"
        return f"http://{self._host_with_optional_port(host)}{self._random_path()}{self._random_query()}"

    @staticmethod
    def _random_path() -> str:
        fragments = ['login', 'verify', 'auth', 'account', 'payment', 'security', 'docs']
        return "/" + "/".join(random.sample(fragments, k=random.randint(1, 3)))

    @staticmethod
    def _random_query() -> str:
        params = [
            f"session={random.randint(100000, 999999)}",
            f"redirect={random.choice(['home', 'verify', 'profile', 'billing'])}",
            f"token={''.join(random.choices('abcdef0123456789', k=16))}",
        ]
        if random.random() < 0.6:
            return "?" + "&".join(random.sample(params, k=random.randint(1, 3)))
        return ""

    @staticmethod
    def _host_with_optional_port(host: str) -> str:
        if random.random() < 0.2:
            return f"{host}:{random.choice([8080, 8081, 8443, 9001, 1337])}"
        return host


class TimeSeriesGenerator(DynamicGenerator):
    """Generate timeseries attack patterns with difficulty-based gradual attacks."""

    DEFAULT_MALICIOUS_WEIGHTS = {
        'ddos': 0.30,
        'portscan': 0.25,
        'exfiltration': 0.20,
        'c2': 0.15,
        'bruteforce': 0.10
    }
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        from .difficulty import DifficultyMixin
        self.difficulty_mixin = DifficultyMixin()
    
    def generate(self, count: int, category_weights: Dict[str, float], benign_ratio: float = 0.7) -> List[Scenario]:
        scenarios = []
        benign_count = int(count * benign_ratio)
        malicious_count = count - benign_count
        
        # Generate benign (normal) samples
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        for i in range(benign_count):
            data = self._generate_normal()
            difficulty = random.choice(difficulties)
            scenarios.append(Scenario(
                id=f"timeseries_benign_{i}_{random.randint(1000,9999)}",
                model='timeseries',
                category='normal',
                subcategory='benign',
                input_data=data,
                expected_label=0,
                difficulty=difficulty,
                description='Normal traffic pattern',
                source='dynamic'
            ))
        
        # Generate malicious samples
        normalized_weights = self._normalize_weights(category_weights, self.DEFAULT_MALICIOUS_WEIGHTS)
        categories = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        
        for i in range(malicious_count):
            category = random.choices(categories, weights=weights)[0]
            difficulty = random.choice(difficulties)
            
            if category == 'ddos':
                data = self._generate_ddos(difficulty)
            elif category == 'portscan':
                data = self._generate_portscan(difficulty)
            elif category == 'exfiltration':
                data = self._generate_exfiltration(difficulty)
            elif category == 'c2':
                data = self._generate_c2_beaconing(difficulty)
            elif category == 'bruteforce':
                data = self._generate_bruteforce(difficulty)
            else:
                data = self._generate_generic_attack(difficulty)

            data = self.difficulty_mixin.apply_difficulty(data, difficulty, 'timeseries')
            data = np.clip(data.astype(np.float32), a_min=0.0, a_max=50000.0)
            
            scenarios.append(Scenario(
                id=f"timeseries_mal_{i}_{random.randint(1000,9999)}",
                model='timeseries',
                category=category,
                subcategory='dynamic',
                input_data=data,
                expected_label=1,
                difficulty=difficulty,
                description=f'Dynamic {category} pattern ({difficulty})',
                source='dynamic'
            ))
        return scenarios
    
    def _generate_ddos(self, difficulty: str) -> np.ndarray:
        """Generate DDoS pattern with difficulty-based ramp: [60, 8] array."""
        seq = np.zeros((60, 8), dtype=np.float32)
        attack_start = random.randint(10, 40)
        attack_duration = 60 - attack_start
        
        # Baseline traffic
        seq[:, 0] = 50
        seq[:, 1] = seq[:, 0] * np.random.uniform(500, 800)
        seq[:, 2] = 50
        seq[:, 3] = 0.02
        seq[:, 4:] = np.random.uniform(10, 100, (60, 4))
        
        # Apply attack based on difficulty
        if difficulty == 'easy':
            # Instant spike (original behavior)
            seq[attack_start:, 0] = np.random.uniform(500, 2000, attack_duration)
            seq[attack_start:, 2] = np.random.uniform(1000, 5000, attack_duration)
            seq[attack_start:, 3] = np.random.uniform(0.3, 0.8, attack_duration)
        
        elif difficulty == 'medium':
            # Linear ramp over 10 timesteps
            ramp_duration = min(10, attack_duration)
            ramp = np.linspace(50, 1500, ramp_duration)
            seq[attack_start:attack_start+ramp_duration, 0] = ramp
            seq[attack_start+ramp_duration:, 0] = np.random.uniform(1500, 2000, max(0, attack_duration-ramp_duration))
            seq[attack_start:, 2] = seq[attack_start:, 0] * np.random.uniform(15, 25)
            seq[attack_start:, 3] = np.random.uniform(0.3, 0.6, attack_duration)
        
        elif difficulty == 'hard':
            # Exponential ramp + noise
            ramp_duration = min(15, attack_duration)
            t = np.linspace(0, 3, ramp_duration)
            ramp = 50 + 1450 * (1 - np.exp(-t))
            noise = np.random.normal(0, 100, ramp_duration)
            seq[attack_start:attack_start+ramp_duration, 0] = ramp + noise
            seq[attack_start+ramp_duration:, 0] = np.random.uniform(1200, 1800, max(0, attack_duration-ramp_duration))
            seq[attack_start:, 2] = seq[attack_start:, 0] * np.random.uniform(10, 20)
            seq[attack_start:, 3] = np.random.uniform(0.2, 0.5, attack_duration)
        
        elif difficulty == 'adversarial':
            # Slow-rate attack (barely above threshold)
            slow_increase = np.linspace(50, 200, attack_duration)
            noise = np.random.normal(0, 20, attack_duration)
            seq[attack_start:, 0] = slow_increase + noise
            seq[attack_start:, 2] = seq[attack_start:, 0] * np.random.uniform(8, 12)
            seq[attack_start:, 3] = np.random.uniform(0.05, 0.15, attack_duration)
        
        # Update bytes based on connections
        seq[:, 1] = seq[:, 0] * np.random.uniform(500, 800)
        
        return seq
    
    def _generate_portscan(self, difficulty: str) -> np.ndarray:
        """Generate port scan pattern with difficulty-based ramp."""
        seq = np.zeros((60, 8), dtype=np.float32)
        scan_start = random.randint(5, 30)
        scan_duration = 60 - scan_start
        
        # Baseline
        seq[:, 0] = 50
        seq[:, 1] = seq[:, 0] * 800
        seq[:, 2] = 50
        seq[:, 4:] = np.random.uniform(10, 100, (60, 4))
        
        # Apply scan based on difficulty
        if difficulty == 'easy':
            seq[scan_start:, 0] = np.random.uniform(100, 300, scan_duration)
            seq[scan_start:, 2] = np.random.uniform(200, 500, scan_duration)
        elif difficulty == 'medium':
            ramp = np.linspace(50, 250, min(8, scan_duration))
            seq[scan_start:scan_start+len(ramp), 0] = ramp
            seq[scan_start+len(ramp):, 0] = np.random.uniform(250, 300, max(0, scan_duration-len(ramp)))
            seq[scan_start:, 2] = seq[scan_start:, 0] * 2
        elif difficulty == 'hard':
            ramp = 50 + 200 * (1 - np.exp(-np.linspace(0, 2, min(12, scan_duration))))
            seq[scan_start:scan_start+len(ramp), 0] = ramp + np.random.normal(0, 15, len(ramp))
            seq[scan_start:, 2] = seq[scan_start:, 0] * np.random.uniform(1.5, 2.5)
        else:  # adversarial
            slow_scan = np.linspace(50, 120, scan_duration)
            seq[scan_start:, 0] = slow_scan + np.random.normal(0, 10, scan_duration)
            seq[scan_start:, 2] = seq[scan_start:, 0] * np.random.uniform(1.2, 1.8)
        
        seq[:, 1] = seq[:, 0] * 800
        return seq
    
    def _generate_normal(self) -> np.ndarray:
        """Generate normal traffic pattern."""
        seq = np.zeros((60, 8), dtype=np.float32)
        t = np.linspace(0, 4*np.pi, 60)
        
        seq[:, 0] = 50 + 20 * np.sin(t) + np.random.normal(0, 3, 60)
        seq[:, 1] = seq[:, 0] * np.random.uniform(800, 1200) + np.random.normal(0, 500, 60)
        seq[:, 2] = np.random.uniform(20, 80) + np.random.normal(0, 5, 60)
        seq[:, 3] = np.clip(np.random.exponential(0.02, 60), 0, 0.2)
        seq[:, 4:] = np.random.uniform(10, 100, (60, 4))
        
        return seq
    
    def _generate_generic_attack(self, difficulty: str) -> np.ndarray:
        """Generate generic attack pattern with difficulty-based ramp."""
        seq = self._generate_normal()
        window = random.randint(15, 45)
        burst = random.randint(6, 15)
        seq[window:window + burst, 0] *= np.random.uniform(1.8, 3.0)
        seq[window:window + burst, 2] *= np.random.uniform(2.0, 4.0)
        return seq.astype(np.float32)

    def _generate_exfiltration(self, difficulty: str) -> np.ndarray:
        """Low-and-slow data exfiltration pattern."""
        seq = self._generate_normal()
        start = random.randint(20, 35)
        ramp = np.linspace(1.0, 1.8 if difficulty == 'adversarial' else 2.8, 60 - start)
        seq[start:, 1] *= ramp  # bytes out grows
        seq[start:, 3] = np.clip(seq[start:, 3] + np.random.uniform(0.02, 0.08, 60 - start), 0, 1)
        return seq.astype(np.float32)

    def _generate_c2_beaconing(self, difficulty: str) -> np.ndarray:
        """Periodic beaconing with low-volume regular traffic."""
        seq = self._generate_normal()
        period = 8 if difficulty in ['easy', 'medium'] else 12
        for t in range(0, 60, period):
            seq[t, 2] = seq[t, 2] * np.random.uniform(2.0, 4.0)
            seq[t, 3] = np.clip(seq[t, 3] + np.random.uniform(0.1, 0.25), 0, 1)
        return seq.astype(np.float32)

    def _generate_bruteforce(self, difficulty: str) -> np.ndarray:
        """Authentication burst and retry cycles."""
        seq = self._generate_normal()
        for _ in range(random.randint(3, 6)):
            start = random.randint(5, 50)
            duration = random.randint(2, 5 if difficulty == 'adversarial' else 8)
            seq[start:start + duration, 0] *= np.random.uniform(1.8, 3.5)
            seq[start:start + duration, 3] = np.clip(seq[start:start + duration, 3] + np.random.uniform(0.15, 0.35), 0, 1)
        return seq.astype(np.float32)


class TabularGenerator(DynamicGenerator):
    """Generate fraud/host/network feature vectors."""

    DEFAULT_WEIGHTS = {
        'fraud': {'card_not_present': 0.40, 'account_takeover': 0.35, 'synthetic': 0.25},
        'host': {'spyware': 0.25, 'ransomware': 0.25, 'trojan': 0.20, 'rootkit': 0.15, 'backdoor': 0.15},
        'network': {'dos': 0.35, 'probe': 0.30, 'r2l': 0.20, 'u2r': 0.15},
        'anomaly': {'zero_day': 0.35, 'stealth_scan': 0.25, 'low_and_slow_exfiltration': 0.25, 'beaconing': 0.15},
    }

    def __init__(self, seed: int = None):
        super().__init__(seed)
        from .difficulty import DifficultyMixin
        self.difficulty_mixin = DifficultyMixin()
        self.feature_profiles = self._load_feature_profiles()

    @staticmethod
    def _load_feature_profiles() -> Dict[str, Dict]:
        profiles = {}
        base = Path(__file__).parent.parent.parent.parent / 'configs' / 'stress_test' / 'feature_profiles'
        if base.exists():
            for path in base.glob('*.json'):
                if not path.is_file():
                    continue
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    model = data.get('model')
                    if model:
                        profiles[model] = data
                except Exception:
                    continue
        return profiles

    @staticmethod
    def _sample_truncnorm(p01: np.ndarray, p50: np.ndarray, p99: np.ndarray) -> np.ndarray:
        """Sample from a bounded normal distribution using quantile estimates."""
        p01 = np.asarray(p01, dtype=np.float32)
        p50 = np.asarray(p50, dtype=np.float32)
        p99 = np.asarray(p99, dtype=np.float32)
        # Approximate std from quantiles; avoid zero std
        std = np.maximum((p99 - p01) / 4.0, 1e-6)
        sample = np.random.normal(loc=p50, scale=std).astype(np.float32)
        return np.clip(sample, p01, p99)

    def _profile_sample(self, model: str, category: str) -> Optional[np.ndarray]:
        profile = self.feature_profiles.get(model)
        if not profile:
            return None
        categories = profile.get('profiles', {})
        entry = categories.get(category)
        if not entry:
            return None
        feature_order = profile.get('features', [])
        p01 = entry.get('p01')
        p50 = entry.get('p50')
        p99 = entry.get('p99')
        if isinstance(p01, dict) and feature_order:
            p01 = [p01.get(name, 0.0) for name in feature_order]
        if isinstance(p50, dict) and feature_order:
            p50 = [p50.get(name, 0.0) for name in feature_order]
        if isinstance(p99, dict) and feature_order:
            p99 = [p99.get(name, 0.0) for name in feature_order]
        return self._sample_truncnorm(p01, p50, p99)

    def generate(self, model: str, count: int, category_weights: Dict[str, float], benign_ratio: float = 0.7) -> List[Scenario]:
        scenarios = []
        benign_count = int(count * benign_ratio)
        malicious_count = count - benign_count

        default_weights = self.DEFAULT_WEIGHTS.get(model, {'generic_attack': 1.0})
        normalized_weights = self._normalize_weights(category_weights, default_weights)

        # Generate benign samples
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        for i in range(benign_count):
            if model == 'fraud':
                features = self._generate_fraud('normal')
            elif model == 'host':
                sample = self._profile_sample('host', 'normal')
                features = sample if sample is not None else self._generate_host('normal')
            elif model == 'network':
                sample = self._profile_sample('network', 'normal')
                features = sample if sample is not None else self._generate_network('normal')
            elif model == 'anomaly':
                features = self._generate_anomaly('normal')
            else:
                features = np.random.randn(10).astype(np.float32)

            difficulty = random.choice(difficulties)
            features = self.difficulty_mixin.apply_difficulty(features, difficulty, 'tabular').astype(np.float32)
            scenarios.append(Scenario(
                id=f"{model}_benign_{i}_{random.randint(1000,9999)}",
                model=model,
                category='normal',
                subcategory='benign',
                input_data=features,
                expected_label=0,
                difficulty=difficulty,
                description='Normal sample',
                source='dynamic'
            ))

        # Generate malicious samples
        categories = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
        
        for i in range(malicious_count):
            category = random.choices(categories, weights=weights)[0]
            difficulty = random.choice(difficulties)

            if model == 'fraud':
                features = self._generate_fraud(category)
            elif model == 'host':
                sample = self._profile_sample('host', category)
                features = sample if sample is not None else self._generate_host(category)
            elif model == 'network':
                sample = self._profile_sample('network', category)
                features = sample if sample is not None else self._generate_network(category)
            elif model == 'anomaly':
                features = self._generate_anomaly(category)
            else:
                features = np.random.randn(10).astype(np.float32)

            features = self.difficulty_mixin.apply_difficulty(features, difficulty, 'tabular').astype(np.float32)
            scenarios.append(Scenario(
                id=f"{model}_mal_{i}_{random.randint(1000,9999)}",
                model=model,
                category=category,
                subcategory='dynamic',
                input_data=features,
                expected_label=1,
                difficulty=difficulty,
                description=f'Dynamic {category} sample',
                source='dynamic'
            ))
        return scenarios
    
    def _generate_fraud(self, category: str) -> np.ndarray:
        """Generate 30-element fraud feature vector with realistic distributions."""
        features = np.zeros(30, dtype=np.float32)
        features[0] = random.uniform(0, 172800)  # Time (0-2 days in seconds)
        
        if category == 'normal':
            features[1:29] = np.random.normal(0, 1.5, 28)
            features[29] = max(1, np.random.lognormal(3.5, 1.2))
        elif category == 'card_not_present':
            features[1:29] = np.random.normal(0.8, 3.0, 28)
            features[29] = max(1, np.random.lognormal(5.0, 1.2))
            features[10:15] += np.random.uniform(2.0, 6.0, 5)
        elif category == 'account_takeover':
            features[1:29] = np.random.normal(-0.5, 3.2, 28)
            features[29] = max(1, np.random.lognormal(4.2, 1.4))
            features[5:10] += np.random.uniform(3.0, 8.0, 5)
        elif category == 'synthetic':
            features[1:29] = np.random.normal(0, 4.0, 28)
            outlier_indices = np.random.choice(28, size=7, replace=False)
            features[1 + outlier_indices] = np.random.uniform(-18, 18, 7)
            features[29] = max(1, np.random.lognormal(5.3, 1.7))
        else:
            features[1:29] = np.random.normal(0, 3.5, 28)
            outlier_indices = np.random.choice(28, size=5, replace=False)
            features[1 + outlier_indices] = np.random.uniform(-15, 15, 5)
            features[29] = max(1, np.random.lognormal(4.5, 1.5))
        
        return features
    
    def _generate_host(self, category: str) -> np.ndarray:
        """Generate 37-element host behavior vector."""
        features = np.zeros(37, dtype=np.float32)
        
        if category == 'normal':
            features[:10] = np.random.uniform(50, 150, 10)
            features[10:] = np.random.uniform(0, 80, 27)
        elif category == 'ransomware':
            features[:10] = np.random.uniform(220, 650, 10)
            features[10:20] = np.random.uniform(60, 100, 10)
            features[20:] = np.random.uniform(20, 120, 17)
        elif category == 'spyware':
            features[:10] = np.random.uniform(120, 300, 10)
            features[10:20] = np.random.uniform(10, 60, 10)
            features[20:] = np.random.uniform(40, 140, 17)
        elif category == 'trojan':
            features[:10] = np.random.uniform(140, 360, 10)
            features[10:20] = np.random.uniform(30, 90, 10)
            features[20:] = np.random.uniform(35, 125, 17)
        elif category == 'rootkit':
            features[:10] = np.random.uniform(180, 420, 10)
            features[10:20] = np.random.uniform(50, 100, 10)
            features[20:] = np.random.uniform(30, 110, 17)
        elif category == 'backdoor':
            features[:10] = np.random.uniform(130, 340, 10)
            features[10:20] = np.random.uniform(20, 75, 10)
            features[20:] = np.random.uniform(45, 130, 17)
        else:
            features[:10] = np.random.uniform(100, 500, 10)
            features[10:] = np.random.uniform(0, 100, 27)
        return features
    
    def _generate_network(self, category: str) -> np.ndarray:
        """Generate 35-element network intrusion vector."""
        features = np.zeros(35, dtype=np.float32)
        
        if category == 'normal':
            features[0] = random.randint(0, 1000)  # duration
            features[1:3] = np.random.uniform(100, 10000, 2)  # bytes
            features[3:] = np.random.uniform(0.0, 0.35, 32)
        elif category == 'dos':
            features[0] = random.uniform(0, 20)
            features[1:3] = np.random.uniform(0, 1200, 2)
            features[3:] = np.random.uniform(0.6, 1.0, 32)
        elif category == 'probe':
            features[0] = random.uniform(0, 80)
            features[1:3] = np.random.uniform(50, 4000, 2)
            features[3:] = np.random.uniform(0.35, 0.85, 32)
        elif category == 'r2l':
            features[0] = random.uniform(20, 300)
            features[1:3] = np.random.uniform(10, 1800, 2)
            features[3:] = np.random.uniform(0.4, 0.95, 32)
        elif category == 'u2r':
            features[0] = random.uniform(5, 160)
            features[1:3] = np.random.uniform(0, 900, 2)
            features[3:] = np.random.uniform(0.5, 1.0, 32)
        else:
            features[0] = 0
            features[1:3] = np.random.uniform(0, 1000, 2)
            features[3:] = np.random.uniform(0.0, 1.0, 32)
        return features

    def _generate_anomaly(self, category: str) -> np.ndarray:
        """Generate 15-feature anomaly vector aligned to UNIFIED_FEATURES-like shape."""
        features = np.zeros(15, dtype=np.float32)
        if category == 'normal':
            features[:] = np.random.uniform(0.05, 0.35, 15)
        elif category == 'zero_day':
            features[:] = np.random.uniform(0.6, 1.0, 15)
            features[0] = np.random.uniform(0.0, 0.1)
        elif category == 'stealth_scan':
            features[:] = np.random.uniform(0.25, 0.7, 15)
            features[3:6] = np.random.uniform(0.7, 1.0, 3)
        elif category == 'low_and_slow_exfiltration':
            features[:] = np.random.uniform(0.15, 0.55, 15)
            features[1:3] = np.random.uniform(0.75, 1.0, 2)
            features[13:15] = np.random.uniform(0.5, 0.9, 2)
        elif category == 'beaconing':
            features[:] = np.random.uniform(0.2, 0.6, 15)
            features[7:10] = np.random.uniform(0.7, 0.95, 3)
        else:
            features[:] = np.random.uniform(0.35, 0.85, 15)
        return features


class AnomalyGenerator(TabularGenerator):
    """Dedicated generator for anomaly-detector scenarios."""

    def generate(self, model: str, count: int, category_weights: Dict[str, float], benign_ratio: float = 0.7) -> List[Scenario]:
        return super().generate('anomaly', count, category_weights, benign_ratio=benign_ratio)


class MetaGenerator(DynamicGenerator):
    """Generate meta-classifier inputs from real base-model score distributions."""

    DEFAULT_MALICIOUS_WEIGHTS = {'combined': 1.0}
    MODEL_NAMES = ['payload', 'url', 'timeseries', 'network', 'host']
    DIFFICULTY_NOISE = {
        'easy': 0.05,
        'medium': 0.12,
        'hard': 0.22,
        'adversarial': 0.35,
    }
    DEFAULT_DISTRIBUTIONS = {
        'payload': {'benign': {'mean': 0.30, 'std': 0.20}, 'attack': {'mean': 0.85, 'std': 0.15}},
        'url': {'benign': {'mean': 0.25, 'std': 0.20}, 'attack': {'mean': 0.80, 'std': 0.15}},
        'timeseries': {'benign': {'mean': 0.10, 'std': 0.12}, 'attack': {'mean': 0.75, 'std': 0.22}},
        'network': {'benign': {'mean': 0.10, 'std': 0.10}, 'attack': {'mean': 0.85, 'std': 0.12}},
        'host': {'benign': {'mean': 0.05, 'std': 0.08}, 'attack': {'mean': 0.90, 'std': 0.08}},
    }

    def __init__(self, seed: int = None, distributions_path: str = 'configs/score_distributions.json'):
        super().__init__(seed)
        self.distributions_path = Path(distributions_path)
        if not self.distributions_path.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            self.distributions_path = project_root / self.distributions_path
        self.score_distributions = self._load_score_distributions()
    
    def generate(self, count: int, category_weights: Dict[str, float], benign_ratio: float = 0.7) -> List[Scenario]:
        """Generate 5-element vectors simulating outputs from 6 base models."""
        scenarios = []
        benign_count = int(count * benign_ratio)
        malicious_count = count - benign_count
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        
        # Generate benign samples
        for i in range(benign_count):
            difficulty = random.choice(difficulties)
            features = self._sample_meta_vector(label=0, difficulty=difficulty)
            scenarios.append(Scenario(
                id=f"meta_benign_{i}_{random.randint(1000,9999)}",
                model='meta',
                category='normal',
                subcategory='benign',
                input_data=features,
                expected_label=0,
                difficulty=difficulty,
                description='Normal meta ensemble',
                source='dynamic'
            ))
        
        # Generate malicious samples
        normalized_weights = self._normalize_weights(category_weights, self.DEFAULT_MALICIOUS_WEIGHTS)
        categories = list(normalized_weights.keys())
        weights = list(normalized_weights.values())
        
        for i in range(malicious_count):
            category = random.choices(categories, weights=weights)[0]
            difficulty = random.choice(difficulties)
            features = self._sample_meta_vector(label=1, difficulty=difficulty)
            scenarios.append(Scenario(
                id=f"meta_mal_{i}_{random.randint(1000,9999)}",
                model='meta',
                category=category,
                subcategory='dynamic',
                input_data=features,
                expected_label=1,
                difficulty=difficulty,
                description=f'Dynamic meta ensemble {category}',
                source='dynamic'
            ))
        
        return scenarios

    def _load_score_distributions(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        distributions = {}
        if self.distributions_path.exists():
            try:
                with open(self.distributions_path, encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    distributions = loaded
            except (OSError, ValueError):
                distributions = {}

        merged = {}
        for model_name in self.MODEL_NAMES:
            default = self.DEFAULT_DISTRIBUTIONS[model_name]
            current = distributions.get(model_name, {})
            merged[model_name] = {
                'benign': {
                    'mean': float(current.get('benign', {}).get('mean', default['benign']['mean'])),
                    'std': float(current.get('benign', {}).get('std', default['benign']['std'])),
                    'p10': float(current.get('benign', {}).get('p10', 0.01)),
                    'p90': float(current.get('benign', {}).get('p90', 0.99)),
                },
                'attack': {
                    'mean': float(current.get('attack', {}).get('mean', default['attack']['mean'])),
                    'std': float(current.get('attack', {}).get('std', default['attack']['std'])),
                    'p10': float(current.get('attack', {}).get('p10', 0.01)),
                    'p90': float(current.get('attack', {}).get('p90', 0.99)),
                },
            }
        return merged

    def _sample_base_score(self, model_name: str, expected_label: int) -> float:
        dist_key = 'attack' if expected_label == 1 else 'benign'
        dist = self.score_distributions[model_name][dist_key]
        raw = np.random.normal(dist['mean'], max(dist['std'], 1e-3))
        bounded = np.clip(raw, dist.get('p10', 0.01), dist.get('p90', 0.99))
        return float(np.clip(bounded, 0.01, 0.99))

    def _sample_meta_vector(self, label: int, difficulty: str) -> np.ndarray:
        """Sample model-score vectors using observed score distributions."""
        noise_std = self.DIFFICULTY_NOISE.get(difficulty, 0.12)
        scores = []
        for model_name in self.MODEL_NAMES:
            base = self._sample_base_score(model_name, label)
            score = base + np.random.normal(0.0, noise_std)
            scores.append(float(np.clip(score, 0.01, 0.99)))

        if difficulty == 'adversarial':
            disagree_idx = np.random.choice(len(self.MODEL_NAMES), 2, replace=False)
            for idx in disagree_idx:
                flipped = 1.0 - scores[idx] + np.random.normal(0.0, 0.05)
                scores[idx] = float(np.clip(flipped, 0.01, 0.99))

        return np.array(scores, dtype=np.float32)


class BenignAdversarialGenerator(DynamicGenerator):
    """Generate benign samples that look suspicious (false positive testing)."""
    
    SQL_LIKE_TEMPLATES = [
        "SELECT * FROM menu WHERE price < {price}",
        "SELECT name FROM products WHERE category = '{cat}'",
        "UPDATE cart SET quantity = {qty} WHERE id = {id}",
        "DELETE FROM wishlist WHERE user_id = {uid}",
        "INSERT INTO orders (user, total) VALUES ('{user}', {total})",
    ]
    
    CODE_TEMPLATES = [
        "if (x < {n}) {{ alert('{msg}'); }}",
        "for (i = 0; i < {n}; i++) {{ console.log(i); }}",
        "while (count < {n}) {{ count++; }}",
        "function test() {{ return x && y || z; }}",
        "const result = a < b ? 'less' : 'greater';",
    ]
    
    MATH_EXPRESSIONS = [
        "<3 love this",
        "x < 3 and y > 5",
        "if a < b then c",
        "3 < x < 7",
        "score <= 100",
    ]
    
    LEGITIMATE_URLS = [
        "http://paypa1-support.example.com",  # Typo but benign domain
        "http://g00gle-analytics.example.org",
        "http://micros0ft-updates.example.net",
        "http://amaz0n-deals.example.io",
        "http://app1e-store.example.co",
    ]
    
    def generate(self, count: int, category_weights: Dict[str, float] = None) -> List[Scenario]:
        """Generate benign adversarial scenarios."""
        scenarios = []
        
        for i in range(count):
            # Choose pattern type
            pattern_type = random.choice(['sql_like', 'code', 'math', 'url'])
            
            if pattern_type == 'sql_like':
                template = random.choice(self.SQL_LIKE_TEMPLATES)
                data = template.format(
                    price=random.randint(5, 50),
                    cat=random.choice(['electronics', 'books', 'clothing']),
                    qty=random.randint(1, 10),
                    id=random.randint(1, 1000),
                    user=random.choice(['john', 'jane', 'admin']),
                    total=random.randint(10, 500)
                )
                model = 'payload'
                category = 'benign_sql_like'
                
            elif pattern_type == 'code':
                template = random.choice(self.CODE_TEMPLATES)
                data = template.format(
                    n=random.randint(1, 10),
                    msg=random.choice(['hello', 'done', 'success'])
                )
                model = 'payload'
                category = 'benign_code'
                
            elif pattern_type == 'math':
                data = random.choice(self.MATH_EXPRESSIONS)
                model = 'payload'
                category = 'benign_math'
                
            else:  # url
                data = random.choice(self.LEGITIMATE_URLS)
                model = 'url'
                category = 'benign_typo_url'
            
            scenarios.append(Scenario(
                id=f"benign_adv_{i}_{random.randint(1000,9999)}",
                model=model,
                category=category,
                subcategory='adversarial_benign',
                input_data=data,
                expected_label=0,  # Should be classified as benign
                difficulty='adversarial',
                description=f'Benign but suspicious: {pattern_type}',
                source='dynamic'
            ))
        
        return scenarios
