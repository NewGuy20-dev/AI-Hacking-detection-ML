"""Real attack data loader for stress testing."""
import random
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


class RealDataLoader:
    """Load and sample from actual attack datasets."""
    
    def __init__(self, base_path: Optional[Path] = None):
        """Initialize loader with base dataset path."""
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent.parent / 'datasets'
        self.base_path = Path(base_path)
        
        # Cache for loaded data
        self._payload_cache: Dict[str, List[str]] = {}
        self._url_cache: Dict[str, List[str]] = {}
        self._benign_cache: List[str] = []
        
        self._load_all()
    
    def _load_all(self):
        """Load all datasets into memory."""
        self._load_payloads()
        self._load_urls()
        self._load_benign_adversarial()
    
    def _load_payloads(self):
        """Load PayloadsAllTheThings attack samples."""
        payload_dir = self.base_path / 'security_payloads' / 'PayloadsAllTheThings'
        
        # SQL Injection
        sql_files = [
            'SQL Injection/Intruder/Auth_Bypass.txt',
            'SQL Injection/Intruder/Auth_Bypass2.txt',
        ]
        self._payload_cache['sqli'] = self._load_text_files(payload_dir, sql_files, fallback_sqli=True)
        
        # XSS
        xss_files = [
            'XSS Injection/Intruder/xss-payload-list.txt',
        ]
        self._payload_cache['xss'] = self._load_text_files(payload_dir, xss_files, fallback_xss=True)
        
        # Command Injection
        cmd_files = [
            'Command Injection/Intruder/command-execution-unix.txt',
            'Command Injection/Intruder/command_exec.txt',
        ]
        self._payload_cache['cmdi'] = self._load_text_files(payload_dir, cmd_files, fallback_cmdi=True)
        
        # Path Traversal
        path_files = [
            'Directory Traversal/Intruder/dotdotpwn.txt',
        ]
        self._payload_cache['path'] = self._load_text_files(payload_dir, path_files, fallback_path=True)
        
        # SSTI
        self._payload_cache['ssti'] = [
            "{{7*7}}", "{{config}}", "{{request}}", "${7*7}", 
            "{{''.__class__.__mro__[1].__subclasses__()}}", "{{config.items()}}",
            "<%= 7*7 %>", "${7*7}", "#{7*7}", "*{7*7}"
        ]
        
        # XXE
        self._payload_cache['xxe'] = [
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
            '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "http://evil.com/xxe">]><foo>&xxe;</foo>',
        ]
        
        # LDAP
        self._payload_cache['ldap'] = [
            '*', '*)(&', '*)(|(&', '*()|(&', 'admin*', 'admin*)((|userPassword=*',
        ]
    
    def _load_text_files(self, base_dir: Path, files: List[str], **fallbacks) -> List[str]:
        """Load text files, return fallback if files don't exist."""
        samples = []
        for file in files:
            path = base_dir / file
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        samples.extend(lines[:100])  # Limit per file
                except Exception:
                    pass
        
        # Use fallbacks if no files loaded
        if not samples:
            if fallbacks.get('fallback_sqli'):
                samples = [
                    "' OR '1'='1", "' OR 1=1--", "admin' --", "' UNION SELECT NULL--",
                    "1' AND '1'='1", "' OR 'a'='a", "1' OR '1'='1'--", "admin' OR '1'='1",
                ]
            elif fallbacks.get('fallback_xss'):
                samples = [
                    "<script>alert(1)</script>", "<img src=x onerror=alert(1)>",
                    "<svg onload=alert(1)>", "javascript:alert(1)", "<iframe src=javascript:alert(1)>",
                ]
            elif fallbacks.get('fallback_cmdi'):
                samples = [
                    "; ls", "| cat /etc/passwd", "&& whoami", "`id`", "$(whoami)",
                ]
            elif fallbacks.get('fallback_path'):
                samples = [
                    "../../../etc/passwd", "..\\..\\..\\windows\\system32\\config\\sam",
                    "....//....//....//etc/passwd",
                ]
        
        return samples
    
    def _load_urls(self):
        """Load malicious URL datasets."""
        # URLhaus
        urlhaus_path = self.base_path / 'url_analysis' / 'urlhaus.csv'
        if urlhaus_path.exists():
            try:
                df = pd.read_csv(urlhaus_path, comment='#', on_bad_lines='skip')
                if 'url' in df.columns:
                    self._url_cache['malware'] = df['url'].dropna().head(1000).tolist()
            except Exception:
                pass
        
        # Kaggle malicious URLs
        kaggle_path = self.base_path / 'url_analysis' / 'kaggle_malicious_urls.csv'
        if kaggle_path.exists():
            try:
                df = pd.read_csv(kaggle_path)
                if 'url' in df.columns:
                    malicious = df[df['type'] != 'benign']['url'].dropna().head(1000).tolist()
                    self._url_cache['phishing'] = malicious
            except Exception:
                pass
        
        # Fallback URLs
        if not self._url_cache.get('malware'):
            self._url_cache['malware'] = [
                'http://malware-site.com/payload.exe',
                'http://evil.com/trojan.zip',
            ]
        if not self._url_cache.get('phishing'):
            self._url_cache['phishing'] = [
                'http://paypa1.com/login',
                'http://g00gle.com/verify',
                'http://micros0ft.com/update',
            ]
    
    def _load_benign_adversarial(self):
        """Load benign but suspicious samples."""
        adv_dir = self.base_path / 'curated_benign' / 'adversarial'
        files = ['code_snippets.txt', 'menu_sql_like.txt', 'math_expressions.txt']
        
        for file in files:
            path = adv_dir / file
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        self._benign_cache.extend(lines[:50])
                except Exception:
                    pass
        
        # Fallback benign adversarial
        if not self._benign_cache:
            self._benign_cache = [
                "SELECT * FROM menu WHERE price < 10",
                "if (x < 3) { alert('hi'); }",
                "<3 love this",
                "x = a && b || c",
            ]
    
    def sample(self, category: str, count: int = 1) -> List[str]:
        """
        Sample from loaded datasets.
        
        Args:
            category: 'sqli', 'xss', 'cmdi', 'path', 'ssti', 'xxe', 'ldap', 
                     'malware_url', 'phishing_url', 'benign_adversarial'
            count: Number of samples to return
        
        Returns:
            List of samples (may be fewer than count if dataset is small)
        """
        if category in self._payload_cache:
            samples = self._payload_cache[category]
        elif category == 'malware_url':
            samples = self._url_cache.get('malware', [])
        elif category == 'phishing_url':
            samples = self._url_cache.get('phishing', [])
        elif category == 'benign_adversarial':
            samples = self._benign_cache
        else:
            return []
        
        if not samples:
            return []
        
        return random.choices(samples, k=min(count, len(samples)))
    
    def get_available_categories(self) -> List[str]:
        """Return list of available categories."""
        categories = list(self._payload_cache.keys())
        categories.extend(['malware_url', 'phishing_url', 'benign_adversarial'])
        return categories
