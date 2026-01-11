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
        
        with open(yaml_path) as f:
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
    """Generate payload variations using mutation techniques."""
    
    BASE_PAYLOADS = {
        'sqli': ["' OR '1'='1", "' UNION SELECT NULL--", "'; DROP TABLE users--", "' AND 1=1--"],
        'xss': ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "<svg onload=alert(1)>"],
        'cmdi': ["| cat /etc/passwd", "; ls -la", "$(whoami)", "`id`"],
        'path_traversal': ["../../../etc/passwd", "..\\..\\..\\windows\\system32\\config\\sam"],
        'ssti': ["{{7*7}}", "${7*7}", "{{config.items()}}"],
        'xxe': ["<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><foo>&xxe;</foo>"],
        'ldap': ["*)(uid=*))(|(uid=*", "admin)(&)"],
    }
    
    def generate(self, count: int, category_weights: Dict[str, float]) -> List[Scenario]:
        scenarios = []
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        
        for i in range(count):
            category = random.choices(categories, weights=weights)[0]
            base = random.choice(self.BASE_PAYLOADS.get(category, ["test"]))
            mutated = self._mutate(base)
            
            scenarios.append(Scenario(
                id=f"payload_dyn_{i}_{random.randint(1000,9999)}",
                model='payload',
                category=category,
                subcategory='dynamic',
                input_data=mutated,
                expected_label=1,
                difficulty='medium',
                description=f'Dynamic {category} variant',
                source='dynamic'
            ))
        return scenarios
    
    def _mutate(self, payload: str) -> str:
        """Apply random mutations."""
        mutations = [
            lambda p: p,  # No mutation
            lambda p: urllib.parse.quote(p),
            lambda p: p.replace(' ', '/**/'),
            lambda p: ''.join(c.upper() if random.random() > 0.5 else c for c in p),
            lambda p: p.replace("'", "''"),
        ]
        return random.choice(mutations)(payload)


class URLGenerator(DynamicGenerator):
    """Generate URL variations."""
    
    BRANDS = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'netflix']
    TLDS = ['.com', '.net', '.org', '.co', '.io', '.info']
    
    def generate(self, count: int, category_weights: Dict[str, float]) -> List[Scenario]:
        scenarios = []
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        
        for i in range(count):
            category = random.choices(categories, weights=weights)[0]
            
            if category == 'phishing':
                url = self._generate_phishing()
            elif category == 'typosquatting':
                url = self._generate_typosquatting()
            elif category == 'dga':
                url = self._generate_dga()
            else:
                url = self._generate_generic(category)
            
            scenarios.append(Scenario(
                id=f"url_dyn_{i}_{random.randint(1000,9999)}",
                model='url',
                category=category,
                subcategory='dynamic',
                input_data=url,
                expected_label=1,
                difficulty='medium',
                description=f'Dynamic {category} URL',
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
    """Generate timeseries attack patterns."""
    
    def generate(self, count: int, category_weights: Dict[str, float]) -> List[Scenario]:
        scenarios = []
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        
        for i in range(count):
            category = random.choices(categories, weights=weights)[0]
            
            if category == 'ddos':
                data = self._generate_ddos()
            elif category == 'portscan':
                data = self._generate_portscan()
            elif category == 'normal':
                data = self._generate_normal()
            else:
                data = self._generate_generic_attack()
            
            scenarios.append(Scenario(
                id=f"timeseries_dyn_{i}_{random.randint(1000,9999)}",
                model='timeseries',
                category=category,
                subcategory='dynamic',
                input_data=data,
                expected_label=0 if category == 'normal' else 1,
                difficulty='medium',
                description=f'Dynamic {category} pattern',
                source='dynamic'
            ))
        return scenarios
    
    def _generate_ddos(self) -> np.ndarray:
        """Generate DDoS pattern: [60, 8] array."""
        seq = np.zeros((60, 8), dtype=np.float32)
        attack_start = random.randint(10, 40)
        
        seq[:, 0] = 50
        seq[attack_start:, 0] = np.random.uniform(500, 2000, 60-attack_start)
        seq[:, 1] = seq[:, 0] * np.random.uniform(500, 800)
        seq[:, 2] = 50
        seq[attack_start:, 2] = np.random.uniform(1000, 5000, 60-attack_start)
        seq[:, 3] = 0.02
        seq[attack_start:, 3] = np.random.uniform(0.3, 0.8, 60-attack_start)
        seq[:, 4:] = np.random.uniform(10, 100, (60, 4))
        
        return seq
    
    def _generate_portscan(self) -> np.ndarray:
        """Generate port scan pattern."""
        seq = np.zeros((60, 8), dtype=np.float32)
        scan_start = random.randint(5, 30)
        
        seq[:, 0] = 50
        seq[scan_start:, 0] = np.random.uniform(100, 300, 60-scan_start)
        seq[:, 1] = seq[:, 0] * 800
        seq[:, 2] = 50
        seq[scan_start:, 2] = np.random.uniform(200, 500, 60-scan_start)
        seq[:, 4:] = np.random.uniform(10, 100, (60, 4))
        
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
    
    def _generate_generic_attack(self) -> np.ndarray:
        """Generate generic attack pattern."""
        return np.random.randn(60, 8).astype(np.float32) * 50 + 100


class TabularGenerator(DynamicGenerator):
    """Generate fraud/host/network feature vectors."""
    
    def generate(self, model: str, count: int, category_weights: Dict[str, float]) -> List[Scenario]:
        scenarios = []
        categories = list(category_weights.keys())
        weights = list(category_weights.values())
        
        for i in range(count):
            category = random.choices(categories, weights=weights)[0]
            
            if model == 'fraud':
                features = self._generate_fraud(category)
                n_features = 30
            elif model == 'host':
                features = self._generate_host(category)
                n_features = 37
            elif model == 'network':
                features = self._generate_network(category)
                n_features = 35
            else:
                features = np.random.randn(10).astype(np.float32)
                n_features = 10
            
            scenarios.append(Scenario(
                id=f"{model}_dyn_{i}_{random.randint(1000,9999)}",
                model=model,
                category=category,
                subcategory='dynamic',
                input_data=features,
                expected_label=0 if category == 'normal' else 1,
                difficulty='medium',
                description=f'Dynamic {category} sample',
                source='dynamic'
            ))
        return scenarios
    
    def _generate_fraud(self, category: str) -> np.ndarray:
        """Generate 30-element fraud feature vector."""
        features = np.zeros(30, dtype=np.float32)
        features[0] = random.uniform(0, 172800)  # Time
        
        if category == 'normal':
            features[1:29] = np.random.normal(0, 1, 28)  # V1-V28
            features[29] = max(0, np.random.exponential(80))  # Amount
        else:
            features[1:29] = np.random.normal(0, 2, 28)  # Anomalous
            features[29] = max(0, np.random.exponential(150))  # Higher amount
        
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

