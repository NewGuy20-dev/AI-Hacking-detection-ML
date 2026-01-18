#!/usr/bin/env python3
"""Generate synthetic malicious URLs for training."""
import json
import random
from pathlib import Path

# Phishing patterns
PHISHING_DOMAINS = [
    "paypal-secure-login.com", "amazon-account-verify.net", "microsoft-update-security.com",
    "apple-id-locked.com", "facebook-security-check.net", "google-account-suspended.com",
    "netflix-billing-update.com", "bank-of-america-verify.com", "chase-secure-login.net",
    "wells-fargo-alert.com", "citibank-security.net", "usbank-verify.com"
]

# Typosquatting
TYPOSQUAT = [
    "gooogle.com", "faceboook.com", "amazom.com", "paypa1.com", "micros0ft.com",
    "app1e.com", "netf1ix.com", "twiter.com", "instagramm.com", "linkedln.com"
]

# Suspicious TLDs
SUSPICIOUS_TLDS = [".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".work", ".click"]

# Malicious paths
MALICIOUS_PATHS = [
    "/login.php?redirect=http://evil.com",
    "/verify-account?token=",
    "/secure/update-billing.php",
    "/confirm-identity?id=",
    "/reset-password?email=",
    "/../../../etc/passwd",
    "/admin/../../config.php"
]

def generate_malicious_urls(count: int = 100000):
    """Generate synthetic malicious URLs."""
    urls = []
    
    for _ in range(count):
        url_type = random.choice(['phishing', 'typosquat', 'suspicious', 'path_traversal'])
        
        if url_type == 'phishing':
            domain = random.choice(PHISHING_DOMAINS)
            path = random.choice(MALICIOUS_PATHS)
            url = f"http://{domain}{path}{random.randint(1000, 9999)}"
        
        elif url_type == 'typosquat':
            domain = random.choice(TYPOSQUAT)
            url = f"https://{domain}/login"
        
        elif url_type == 'suspicious':
            domain = f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))}{random.choice(SUSPICIOUS_TLDS)}"
            url = f"http://{domain}/download.exe"
        
        else:  # path_traversal
            domain = "legitimate-site.com"
            path = random.choice(MALICIOUS_PATHS)
            url = f"https://{domain}{path}"
        
        urls.append({"text": url, "source": "synthetic", "type": "malicious_url"})
    
    return urls

if __name__ == "__main__":
    output_dir = Path("datasets/malicious_urls")
    output_dir.mkdir(exist_ok=True)
    
    print("Generating 100k malicious URLs...")
    urls = generate_malicious_urls(100000)
    
    output_file = output_dir / "synthetic_malicious_urls.jsonl"
    with open(output_file, 'w') as f:
        for url in urls:
            f.write(json.dumps(url) + '\n')
    
    print(f"✓ Generated {len(urls)} malicious URLs")
    print(f"✓ Saved to {output_file}")
    print(f"\nSample URLs:")
    for url in random.sample(urls, 5):
        print(f"  {url['text']}")
