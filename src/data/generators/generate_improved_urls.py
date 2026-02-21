"""Phase A3: Improved synthetic URL generator with harder patterns."""
import random
import string
from pathlib import Path


# Real-looking TLDs (not suspicious ones)
LEGIT_TLDS = ['.com', '.org', '.net', '.co', '.io', '.app', '.dev', '.me', '.us', '.uk', '.de', '.fr']
BRANDS = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'netflix', 'bank', 
          'chase', 'wellsfargo', 'citibank', 'dropbox', 'linkedin', 'twitter', 'instagram']

# Homograph/typo mappings
TYPOS = {'a': ['4', '@', 'а'], 'e': ['3', 'е'], 'i': ['1', 'l', '!'], 'o': ['0', 'о'],
         's': ['5', '$'], 'l': ['1', 'I'], 't': ['7', '+'], 'g': ['9', 'q']}


def typosquat(word):
    """Create typosquatted version of word."""
    chars = list(word)
    idx = random.randint(0, len(chars) - 1)
    c = chars[idx].lower()
    if c in TYPOS:
        chars[idx] = random.choice(TYPOS[c])
    elif random.random() < 0.5:
        # Double letter or swap adjacent
        if idx < len(chars) - 1:
            chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return ''.join(chars)


def generate_hard_malicious_urls(n=50000):
    """Generate sophisticated malicious URLs that look legitimate."""
    urls = []
    
    for _ in range(n):
        pattern = random.choice(['typo', 'subdomain', 'path', 'combo', 'lookalike'])
        brand = random.choice(BRANDS)
        tld = random.choice(LEGIT_TLDS)
        proto = random.choice(['http://', 'https://'])
        
        if pattern == 'typo':
            # Single character typo: paypa1.com, arnazon.com
            domain = typosquat(brand) + tld
            path = random.choice(['/login', '/signin', '/account', '/verify', '/secure', '/auth'])
            urls.append(f"{proto}{domain}{path}")
            
        elif pattern == 'subdomain':
            # Subdomain abuse: paypal.secure-login.com, amazon.account.evil.co
            fake_domain = random.choice(['secure-login', 'account-verify', 'signin-help', 'support-center'])
            urls.append(f"{proto}{brand}.{fake_domain}{tld}/")
            
        elif pattern == 'path':
            # Legitimate domain with suspicious path (simulating compromised site)
            legit = random.choice(['blog', 'news', 'shop', 'store', 'cdn']) + str(random.randint(1,99))
            urls.append(f"{proto}{legit}{tld}/{brand}/login.php")
            
        elif pattern == 'combo':
            # Combination: secure-paypal-login.com
            prefix = random.choice(['secure', 'login', 'account', 'verify', 'update', 'confirm'])
            suffix = random.choice(['login', 'verify', 'secure', 'auth', 'help'])
            urls.append(f"{proto}{prefix}-{brand}-{suffix}{tld}/")
            
        else:  # lookalike
            # Similar looking: rnicrosoft.com (rn looks like m)
            lookalikes = {'m': 'rn', 'w': 'vv', 'cl': 'd', 'rn': 'm'}
            modified = brand
            for orig, repl in lookalikes.items():
                if orig in modified and random.random() < 0.5:
                    modified = modified.replace(orig, repl, 1)
                    break
            urls.append(f"{proto}{modified}{tld}/signin")
    
    return urls


def generate_hard_benign_urls(n=50000):
    """Generate diverse legitimate URLs including edge cases."""
    urls = []
    
    # Top domains with realistic paths
    domains = ['google.com', 'youtube.com', 'facebook.com', 'amazon.com', 'wikipedia.org',
               'twitter.com', 'instagram.com', 'linkedin.com', 'github.com', 'reddit.com',
               'stackoverflow.com', 'medium.com', 'nytimes.com', 'bbc.com', 'cnn.com',
               'microsoft.com', 'apple.com', 'netflix.com', 'spotify.com', 'twitch.tv']
    
    paths = ['', '/', '/about', '/contact', '/help', '/support', '/blog', '/news',
             '/products', '/services', '/pricing', '/login', '/signup', '/search',
             '/terms', '/privacy', '/careers', '/press', '/api/v1', '/docs']
    
    for _ in range(n):
        pattern = random.choice(['simple', 'query', 'cdn', 'api', 'security_blog'])
        
        if pattern == 'simple':
            domain = random.choice(domains)
            path = random.choice(paths)
            proto = random.choice(['http://', 'https://'])
            www = random.choice(['', 'www.'])
            urls.append(f"{proto}{www}{domain}{path}")
            
        elif pattern == 'query':
            domain = random.choice(domains)
            path = random.choice(['/search', '/products', '/results'])
            query = f"?q={''.join(random.choices(string.ascii_lowercase, k=5))}&page={random.randint(1,10)}"
            urls.append(f"https://{domain}{path}{query}")
            
        elif pattern == 'cdn':
            # CDN URLs (look suspicious but are benign)
            cdn = random.choice(['cdn', 'static', 'assets', 'media', 'images'])
            domain = random.choice(domains).split('.')[0]
            urls.append(f"https://{cdn}.{domain}.com/v{random.randint(1,3)}/file.js")
            
        elif pattern == 'api':
            domain = random.choice(domains)
            urls.append(f"https://api.{domain}/v{random.randint(1,3)}/users/{random.randint(1000,9999)}")
            
        else:  # security_blog - looks malicious but is benign
            security_sites = ['krebsonsecurity.com', 'threatpost.com', 'bleepingcomputer.com',
                            'securityweek.com', 'darkreading.com', 'hackernews.com']
            urls.append(f"https://{random.choice(security_sites)}/malware-analysis-{random.randint(2020,2024)}")
    
    return urls


def main():
    base_path = Path(__file__).parent.parent
    url_dir = base_path / 'datasets' / 'url_analysis'
    url_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating improved synthetic URLs...")
    
    mal_urls = generate_hard_malicious_urls(50000)
    ben_urls = generate_hard_benign_urls(50000)
    
    with open(url_dir / 'synthetic_malicious_hard.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(mal_urls))
    print(f"  ✓ Saved {len(mal_urls)} hard malicious URLs")
    
    with open(url_dir / 'synthetic_benign_hard.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(ben_urls))
    print(f"  ✓ Saved {len(ben_urls)} hard benign URLs")
    
    print("\nDone! These URLs have harder patterns for better model generalization.")


if __name__ == "__main__":
    main()
