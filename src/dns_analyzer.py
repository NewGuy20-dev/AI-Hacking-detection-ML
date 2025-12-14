"""DNS query analysis for detecting malicious domains."""
import math
import re
from collections import Counter


class DNSAnalyzer:
    """Analyze DNS queries for suspicious patterns."""
    
    SUSPICIOUS_TLDS = {'.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq', '.pw', '.cc', '.club', '.work', '.loan'}
    
    def analyze(self, domain: str) -> dict:
        """Extract features and score a domain."""
        domain = domain.lower().strip()
        
        features = {
            'length': len(domain),
            'entropy': self._entropy(domain),
            'digit_ratio': sum(c.isdigit() for c in domain) / max(len(domain), 1),
            'consonant_ratio': self._consonant_ratio(domain),
            'subdomain_count': domain.count('.'),
            'has_suspicious_tld': any(domain.endswith(tld) for tld in self.SUSPICIOUS_TLDS),
            'has_ip_pattern': bool(re.search(r'\d{1,3}[-_.]\d{1,3}', domain)),
            'max_label_length': max((len(p) for p in domain.split('.')), default=0),
            'has_hex_pattern': bool(re.search(r'[0-9a-f]{8,}', domain)),
            'unique_char_ratio': len(set(domain)) / max(len(domain), 1),
        }
        
        # DGA score (higher = more likely DGA)
        dga_score = (
            0.3 * min(features['entropy'] / 4.5, 1) +
            0.2 * features['digit_ratio'] +
            0.2 * features['consonant_ratio'] +
            0.15 * (1 if features['has_suspicious_tld'] else 0) +
            0.15 * min(features['length'] / 50, 1)
        )
        
        return {
            'domain': domain,
            'features': features,
            'dga_score': round(dga_score, 3),
            'is_suspicious': dga_score > 0.5 or features['has_suspicious_tld']
        }
    
    def _entropy(self, s: str) -> float:
        if not s:
            return 0.0
        prob = [c / len(s) for c in Counter(s).values()]
        return -sum(p * math.log2(p) for p in prob if p > 0)
    
    def _consonant_ratio(self, s: str) -> float:
        consonants = set('bcdfghjklmnpqrstvwxyz')
        letters = [c for c in s.lower() if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c in consonants) / len(letters)
    
    def analyze_batch(self, domains: list) -> list:
        return [self.analyze(d) for d in domains]


if __name__ == '__main__':
    analyzer = DNSAnalyzer()
    
    test_domains = [
        'google.com',
        'facebook.com',
        'xn3kd9fj2k.xyz',
        '192-168-1-1.malware.tk',
        'a1b2c3d4e5f6.suspicious.top',
        'legitimate-business.com',
    ]
    
    print("DNS Analysis Results:")
    for domain in test_domains:
        result = analyzer.analyze(domain)
        status = "⚠️ SUSPICIOUS" if result['is_suspicious'] else "✓ OK"
        print(f"  {domain:35s} DGA={result['dga_score']:.3f} {status}")
