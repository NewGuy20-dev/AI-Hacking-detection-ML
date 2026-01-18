"""Scenario dataclasses and registry for V1.4 stress test suite."""
from dataclasses import dataclass
from typing import Any, Optional, List, Dict
from datetime import datetime
from pathlib import Path
import yaml
import random
import base64
import urllib.parse
import numpy as np


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
            # Skip DYNAMIC placeholders
            if item.get('input') == 'DYNAMIC':
                continue
                
            scenarios.append(Scenario(
                id=item['id'],
                model=model_name,
                category=item['category'],
                subcategory=item['subcategory'],
                input_data=item['input'],
                expected_label=item['expected'],
                difficulty=item['difficulty'],
                description=item['description'],
                source='static'
            ))
        
        return scenarios


# Dynamic Generators

class DynamicGenerator:
    """Base class for dynamic scenario generation."""
    def __init__(self, seed: int = None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate(self, count: int, category_weights: Dict[str, float]) -> List[Scenario]:
        raise NotImplementedError


class PayloadGenerator(DynamicGenerator):
    """Generate payload variations using mutation techniques with real data and difficulty tiers."""
    
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
        from real_data import RealDataLoader
        from difficulty import DifficultyMixin
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
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
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
        from real_data import RealDataLoader
        from difficulty import DifficultyMixin
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
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
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
                elif category == 'dga':
                    url = self._generate_dga()
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
        return f"http://{brand}-{random.choice(keywords)}{random.choice(self.TLDS)}"
    
    def _generate_typosquatting(self) -> str:
        brand = random.choice(self.BRANDS)
        typo = brand[:-1] + random.choice('abcdefghijklmnopqrstuvwxyz')
        return f"http://{typo}{random.choice(self.TLDS)}"
    
    def _generate_dga(self) -> str:
        length = random.randint(8, 16)
        domain = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=length))
        return f"http://{domain}{random.choice(self.TLDS)}"
    
    def _generate_generic(self, category: str) -> str:
        return f"http://malicious-{category}-{random.randint(1000, 9999)}.com"


class TimeSeriesGenerator(DynamicGenerator):
    """Generate timeseries attack patterns with difficulty-based gradual attacks."""
    
    def __init__(self, seed: int = None):
        super().__init__(seed)
        from difficulty import DifficultyMixin
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
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        
        for i in range(malicious_count):
            category = random.choices(categories, weights=weights)[0]
            difficulty = random.choice(difficulties)
            
            if category == 'ddos':
                data = self._generate_ddos(difficulty)
            elif category == 'portscan':
                data = self._generate_portscan(difficulty)
            else:
                data = self._generate_generic_attack(difficulty)
            
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
        return np.random.randn(60, 8).astype(np.float32) * 50 + 100


class TabularGenerator(DynamicGenerator):
    """Generate fraud/host/network feature vectors."""
    
    def generate(self, model: str, count: int, category_weights: Dict[str, float], benign_ratio: float = 0.7) -> List[Scenario]:
        scenarios = []
        benign_count = int(count * benign_ratio)
        malicious_count = count - benign_count
        
        # Generate benign samples
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        for i in range(benign_count):
            if model == 'fraud':
                features = self._generate_fraud('normal')
            elif model == 'host':
                features = self._generate_host('normal')
            elif model == 'network':
                features = self._generate_network('normal')
            else:
                features = np.random.randn(10).astype(np.float32)
            
            difficulty = random.choice(difficulties)
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
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        
        for i in range(malicious_count):
            category = random.choices(categories, weights=weights)[0]
            
            if model == 'fraud':
                features = self._generate_fraud(category)
            elif model == 'host':
                features = self._generate_host(category)
            elif model == 'network':
                features = self._generate_network(category)
            else:
                features = np.random.randn(10).astype(np.float32)
            
            scenarios.append(Scenario(
                id=f"{model}_mal_{i}_{random.randint(1000,9999)}",
                model=model,
                category=category,
                subcategory='dynamic',
                input_data=features,
                expected_label=1,
                difficulty='medium',
                description=f'Dynamic {category} sample',
                source='dynamic'
            ))
        return scenarios
    
    def _generate_fraud(self, category: str) -> np.ndarray:
        """Generate 30-element fraud feature vector with realistic distributions."""
        features = np.zeros(30, dtype=np.float32)
        features[0] = random.uniform(0, 172800)  # Time (0-2 days in seconds)
        
        if category == 'normal':
            # Normal transactions: V1-V28 centered around 0 with small variance
            features[1:29] = np.random.normal(0, 1.5, 28)
            # Amount: typical range $1-$200
            features[29] = max(1, np.random.lognormal(3.5, 1.2))
        else:
            # Fraudulent transactions: more extreme V values
            # Mix of high positive and negative values
            features[1:29] = np.random.normal(0, 3.5, 28)
            # Add some extreme outliers for fraud patterns
            outlier_indices = np.random.choice(28, size=5, replace=False)
            features[1 + outlier_indices] = np.random.uniform(-15, 15, 5)
            # Amount: higher and more variable
            features[29] = max(1, np.random.lognormal(4.5, 1.5))
        
        return features
    
    def _generate_host(self, category: str) -> np.ndarray:
        """Generate 37-element host behavior vector."""
        features = np.zeros(37, dtype=np.float32)
        
        if category == 'normal':
            features[:10] = np.random.uniform(50, 150, 10)
        else:
            features[:10] = np.random.uniform(100, 500, 10)  # Anomalous
        
        features[10:] = np.random.uniform(0, 100, 27)
        return features
    
    def _generate_network(self, category: str) -> np.ndarray:
        """Generate 35-element network intrusion vector."""
        features = np.zeros(35, dtype=np.float32)
        
        if category == 'normal':
            features[0] = random.randint(0, 1000)  # duration
            features[1:3] = np.random.uniform(100, 10000, 2)  # bytes
        else:
            features[0] = 0  # Short duration for attacks
            features[1:3] = np.random.uniform(0, 1000, 2)  # Low bytes
        
        features[3:] = np.random.uniform(0, 1, 32)
        return features


class MetaGenerator(DynamicGenerator):
    """Generate meta-classifier input vectors (simulated model outputs)."""
    
    def generate(self, count: int, category_weights: Dict[str, float], benign_ratio: float = 0.7) -> List[Scenario]:
        """Generate 5-element vectors simulating outputs from 6 base models."""
        scenarios = []
        benign_count = int(count * benign_ratio)
        malicious_count = count - benign_count
        difficulties = ['easy', 'medium', 'hard', 'adversarial']
        
        # Generate benign samples
        for i in range(benign_count):
            # Normal: low probabilities (0.0-0.3)
            features = np.random.uniform(0.0, 0.3, 5).astype(np.float32)
            difficulty = random.choice(difficulties)
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
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        
        for i in range(malicious_count):
            category = random.choices(categories, weights=weights)[0]
            # Attack: high probabilities (0.7-1.0)
            features = np.random.uniform(0.7, 1.0, 5).astype(np.float32)
            scenarios.append(Scenario(
                id=f"meta_mal_{i}_{random.randint(1000,9999)}",
                model='meta',
                category=category,
                subcategory='dynamic',
                input_data=features,
                expected_label=1,
                difficulty='medium',
                description=f'Dynamic meta ensemble {category}',
                source='dynamic'
            ))
        
        return scenarios


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

