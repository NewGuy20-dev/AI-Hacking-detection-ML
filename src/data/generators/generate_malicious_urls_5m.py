#!/usr/bin/env python3
"""Generate 5M synthetic malicious URLs for URL CNN training."""
import json
import random
import string
from pathlib import Path
from tqdm import tqdm

# Malicious patterns
SUSPICIOUS_TLDS = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.loan', '.work', 
                   '.click', '.link', '.info', '.online', '.site', '.club', '.buzz']
PHISHING_KEYWORDS = ['login', 'signin', 'verify', 'secure', 'account', 'update', 'confirm',
                     'banking', 'paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
                     'netflix', 'support', 'help', 'service', 'alert', 'suspended', 'locked']
MALWARE_KEYWORDS = ['download', 'free', 'crack', 'keygen', 'patch', 'serial', 'warez',
                    'torrent', 'hack', 'cheat', 'bot', 'tool', 'generator', 'exploit']
SUSPICIOUS_PATHS = ['/wp-admin', '/admin', '/login', '/signin', '/verify', '/secure',
                    '/update', '/confirm', '/reset', '/recover', '/unlock', '/validate',
                    '/.env', '/.git', '/config', '/backup', '/db', '/sql', '/shell']

def random_string(length, chars=string.ascii_lowercase):
    return ''.join(random.choices(chars, k=length))

def random_hex(length):
    return ''.join(random.choices('0123456789abcdef', k=length))

def generate_phishing_url():
    """Generate phishing URL mimicking legitimate sites."""
    templates = [
        # Typosquatting
        lambda: f"https://{random.choice(['g00gle', 'googIe', 'gooogle', 'google-secure', 'google-login'])}.com/{random.choice(SUSPICIOUS_PATHS)}",
        lambda: f"https://{random.choice(['paypa1', 'paypal-secure', 'paypal-verify', 'paypaI'])}.com/signin?id={random_hex(16)}",
        lambda: f"https://{random.choice(['arnazon', 'amazon-verify', 'amazon-secure', 'arnazon-login'])}.com/ap/signin",
        lambda: f"https://{random.choice(['faceb00k', 'facebook-login', 'fb-secure'])}.com/login.php?id={random_hex(8)}",
        # Subdomain abuse
        lambda: f"https://{random.choice(PHISHING_KEYWORDS)}.{random_string(8)}{random.choice(SUSPICIOUS_TLDS)}/",
        lambda: f"https://www.{random.choice(PHISHING_KEYWORDS)}-{random_string(5)}.{random_string(6)}{random.choice(SUSPICIOUS_TLDS)}",
        lambda: f"https://{random.choice(['secure', 'login', 'verify'])}.{random.choice(['google', 'apple', 'microsoft'])}.{random_string(8)}{random.choice(SUSPICIOUS_TLDS)}",
        # Long random subdomains
        lambda: f"https://{random_string(20)}.{random_string(10)}{random.choice(SUSPICIOUS_TLDS)}/{random.choice(SUSPICIOUS_PATHS)}",
        # IP-based
        lambda: f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/{random.choice(PHISHING_KEYWORDS)}",
        lambda: f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}:{random.choice([8080,8443,3000,4443])}/login",
        # Encoded/obfuscated
        lambda: f"https://{random_string(10)}{random.choice(SUSPICIOUS_TLDS)}/%2e%2e/{random.choice(PHISHING_KEYWORDS)}",
        lambda: f"https://{random_string(8)}.com/{random_hex(32)}/{random.choice(PHISHING_KEYWORDS)}.php",
    ]
    return random.choice(templates)()

def generate_malware_url():
    """Generate malware distribution URL."""
    templates = [
        # Direct download
        lambda: f"http://{random_string(12)}{random.choice(SUSPICIOUS_TLDS)}/{random.choice(MALWARE_KEYWORDS)}/{random_string(8)}.exe",
        lambda: f"http://{random_string(8)}.com/download/{random_hex(16)}.{random.choice(['exe', 'dll', 'scr', 'bat', 'ps1'])}",
        # Fake software
        lambda: f"https://{random.choice(MALWARE_KEYWORDS)}-{random_string(6)}.com/{random.choice(['adobe', 'office', 'windows', 'antivirus'])}-{random.choice(MALWARE_KEYWORDS)}.exe",
        lambda: f"http://free-{random_string(8)}.{random.choice(SUSPICIOUS_TLDS)}/{random.choice(MALWARE_KEYWORDS)}.zip",
        # Exploit kit patterns
        lambda: f"http://{random_string(15)}{random.choice(SUSPICIOUS_TLDS)}/gate.php?id={random_hex(32)}",
        lambda: f"http://{random_string(10)}.com/{random_hex(8)}/{random_hex(8)}/{random_hex(8)}.js",
        lambda: f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/load.php?{random_hex(16)}",
        # C2 patterns
        lambda: f"http://{random_string(20)}{random.choice(SUSPICIOUS_TLDS)}/api/{random_hex(8)}/beacon",
        lambda: f"https://{random_string(12)}.com/update/{random_hex(16)}.bin",
        lambda: f"http://{random_hex(8)}.{random.choice(SUSPICIOUS_TLDS)}/{random_hex(4)}/{random_hex(4)}/{random_hex(4)}",
    ]
    return random.choice(templates)()

def generate_suspicious_url():
    """Generate generally suspicious URL patterns."""
    templates = [
        # Excessive parameters
        lambda: f"https://{random_string(8)}.com/page?{'&'.join([f'{random_string(3)}={random_hex(8)}' for _ in range(random.randint(5,15))])}",
        # Deep paths
        lambda: f"https://{random_string(10)}.com/{'/'.join([random_string(6) for _ in range(random.randint(5,10))])}",
        # Mixed case abuse
        lambda: f"https://{''.join(random.choice([c.upper(), c.lower()]) for c in random_string(15))}.com/{random.choice(SUSPICIOUS_PATHS)}",
        # Special chars
        lambda: f"https://{random_string(8)}.com/{random_string(5)}@{random_string(5)}/{random_hex(8)}",
        lambda: f"https://{random_string(8)}.com/redirect?url=http://{random_string(10)}.com",
        # Base64-like
        lambda: f"https://{random_string(8)}.com/d/{random_string(44).replace('a','A').replace('b','B')}",
        # Punycode-like
        lambda: f"https://xn--{random_string(10)}.com/{random.choice(PHISHING_KEYWORDS)}",
        # Port abuse
        lambda: f"http://{random_string(10)}.com:{random.randint(1024,65535)}/{random_string(8)}",
        # Data exfil patterns
        lambda: f"https://{random_string(8)}.com/api?data={random_hex(64)}",
        lambda: f"https://{random_hex(32)}.{random_string(6)}{random.choice(SUSPICIOUS_TLDS)}/",
    ]
    return random.choice(templates)()

def main():
    output_path = Path(__file__).parent.parent / "datasets" / "url_analysis" / "malicious_urls_5m.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total = 5_000_000
    generators = [
        (generate_phishing_url, 0.4),      # 40% phishing
        (generate_malware_url, 0.35),      # 35% malware
        (generate_suspicious_url, 0.25),   # 25% suspicious
    ]
    
    print(f"Generating {total:,} malicious URLs...")
    print(f"Output: {output_path}")
    
    with open(output_path, 'w') as f:
        for i in tqdm(range(total)):
            r = random.random()
            if r < 0.4:
                url = generate_phishing_url()
            elif r < 0.75:
                url = generate_malware_url()
            else:
                url = generate_suspicious_url()
            
            f.write(json.dumps({"url": url, "label": 1}) + '\n')
    
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"✓ Generated {total:,} malicious URLs ({size_mb:.1f} MB)")

if __name__ == "__main__":
    main()
