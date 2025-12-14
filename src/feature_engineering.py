"""Feature extraction for Network, URL, and Content detectors."""
import numpy as np
import re
import math
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_url_features(url: str) -> dict:
    """Extract features from a URL string."""
    try:
        parsed = urlparse(url if '://' in url else f'http://{url}')
    except:
        parsed = None
    
    domain = parsed.netloc if parsed else url.split('/')[0]
    path = parsed.path if parsed else ''
    
    # Suspicious TLDs
    suspicious_tlds = {'.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq', '.pw', '.cc', '.club'}
    
    features = {
        'url_length': len(url),
        'domain_length': len(domain),
        'path_depth': path.count('/'),
        'special_char_count': sum(1 for c in url if c in '@-_~!$&\'()*+,;='),
        'digit_ratio': sum(c.isdigit() for c in url) / max(len(url), 1),
        'entropy': _shannon_entropy(url),
        'has_ip': 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) else 0,
        'suspicious_tld': 1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0,
        'has_https': 1 if url.startswith('https') else 0,
        'num_subdomains': domain.count('.'),
        'has_port': 1 if ':' in domain.split('.')[-1] else 0,
        'query_length': len(parsed.query) if parsed else 0,
        'num_params': url.count('='),
        'has_at_symbol': 1 if '@' in url else 0,
        'double_slash_redirect': 1 if '//' in path else 0,
    }
    return features


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def extract_url_features_batch(urls: list) -> np.ndarray:
    """Extract features from a list of URLs."""
    features = [extract_url_features(url) for url in urls]
    keys = list(features[0].keys())
    return np.array([[f[k] for k in keys] for f in features], dtype='float32')


def get_url_feature_names() -> list:
    """Return URL feature names."""
    return list(extract_url_features('http://example.com').keys())


class ContentFeatureExtractor:
    """TF-IDF based content feature extractor."""
    
    def __init__(self, max_features: int = 1000):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            dtype=np.float32
        )
        self.fitted = False
    
    def fit(self, texts: list):
        """Fit the vectorizer on training texts."""
        self.vectorizer.fit(texts)
        self.fitted = True
        return self
    
    def transform(self, texts: list) -> np.ndarray:
        """Transform texts to TF-IDF features."""
        if not self.fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: list) -> np.ndarray:
        """Fit and transform in one step."""
        self.fitted = True
        return self.vectorizer.fit_transform(texts).toarray()


def extract_email_features(text: str) -> dict:
    """Extract features from email/content text."""
    text_lower = text.lower()
    
    urgency_words = ['urgent', 'immediately', 'action required', 'verify', 'confirm', 
                     'suspended', 'limited', 'expire', 'click here', 'act now']
    
    features = {
        'text_length': len(text),
        'num_links': len(re.findall(r'https?://', text)),
        'num_urgency_words': sum(1 for w in urgency_words if w in text_lower),
        'has_html': 1 if '<html' in text_lower or '<body' in text_lower else 0,
        'num_exclamation': text.count('!'),
        'num_question': text.count('?'),
        'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'num_dollar_signs': text.count('$'),
    }
    return features


if __name__ == '__main__':
    # Test URL features
    test_urls = [
        'https://www.google.com/search?q=test',
        'http://192.168.1.1/admin/login.php',
        'http://suspicious-site.xyz/free-money.html',
    ]
    
    print("URL Features:")
    for url in test_urls:
        feats = extract_url_features(url)
        print(f"  {url[:40]}... -> entropy={feats['entropy']:.2f}, suspicious={feats['suspicious_tld']}")
    
    # Test content features
    print("\nContent Features:")
    extractor = ContentFeatureExtractor(max_features=100)
    texts = ["Click here to verify your account immediately!", "Normal email content here."]
    X = extractor.fit_transform(texts)
    print(f"  TF-IDF shape: {X.shape}")
