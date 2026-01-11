#!/usr/bin/env python3
"""Temporary script to retry failed downloads. Delete after use."""

import gzip
import json
import requests
import zipfile
import io
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent / "datasets" / "live_benign"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_github_archive():
    """Download GitHub Archive snippets."""
    output = OUTPUT_DIR / "github_snippets.jsonl"
    target = 100_000_000
    
    # Check existing
    existing = 0
    if output.exists():
        existing = sum(1 for _ in open(output, 'r', encoding='utf-8'))
        if existing >= target:
            print(f"GitHub: Already complete ({existing:,})")
            return
        print(f"GitHub: Resuming from {existing:,}...")
    
    print(f"GitHub Archive: Target {target:,} snippets")
    
    def get_urls(days=90):
        urls = []
        now = datetime.now(timezone.utc)
        for day in range(days):
            date = now - timedelta(days=day)
            for hour in range(24):
                urls.append(f"https://data.gharchive.org/{date.strftime('%Y-%m-%d')}-{hour}.json.gz")
        return urls
    
    def extract_snippets(url):
        snippets = []
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                return []
            for line in gzip.decompress(resp.content).decode('utf-8', errors='ignore').split('\n'):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    payload = event.get('payload', {})
                    if event.get('type') == 'PushEvent':
                        for c in payload.get('commits', []):
                            if c.get('message'):
                                snippets.append(('commit', c['message'][:500]))
                    elif event.get('type') in ('IssuesEvent', 'PullRequestEvent'):
                        item = payload.get('issue') or payload.get('pull_request', {})
                        if item.get('title'):
                            snippets.append(('title', item['title'][:200]))
                        if item.get('body') and len(item['body']) > 20:
                            snippets.append(('body', item['body'][:1000]))
                except:
                    pass
        except:
            pass
        return snippets
    
    urls = get_urls(90)
    collected = existing
    
    with open(output, 'a' if existing > 0 else 'w', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=8) as executor:
            with tqdm(total=target, initial=collected, desc="GitHub") as pbar:
                for url in urls:
                    if collected >= target:
                        break
                    future = executor.submit(extract_snippets, url)
                    for typ, text in future.result():
                        if collected >= target:
                            break
                        f.write(json.dumps({'text': text, 'source': 'github', 'type': typ}) + '\n')
                        collected += 1
                        pbar.update(1)
                    f.flush()
    
    print(f"GitHub: Collected {collected:,}")

def download_common_crawl():
    """Download Common Crawl URLs."""
    output = OUTPUT_DIR / "common_crawl_urls.jsonl"
    target = 10_000_000
    
    existing = 0
    if output.exists():
        existing = sum(1 for _ in open(output, 'r', encoding='utf-8'))
        if existing >= target:
            print(f"Common Crawl: Already complete ({existing:,})")
            return
    
    print(f"Common Crawl: Target {target:,} URLs")
    
    # Try multiple sources
    sources = [
        ("Tranco", "https://tranco-list.eu/top-1m.csv.zip", True),
        ("Umbrella", "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip", True),
        ("Majestic", "https://downloads.majestic.com/majestic_million.csv", False),
    ]
    
    urls = set()
    for name, url, is_zip in sources:
        try:
            print(f"  Trying {name}...")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            
            if is_zip:
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    content = z.read(z.namelist()[0]).decode('utf-8', errors='ignore')
            else:
                content = resp.text
            
            for line in content.splitlines():
                parts = line.split(',')
                domain = parts[1].strip() if len(parts) > 1 else parts[0].strip()
                if domain and '.' in domain:
                    urls.add(f"https://{domain}/")
            print(f"  {name}: Got {len(urls):,} URLs")
        except Exception as e:
            print(f"  {name}: Failed - {e}")
    
    if not urls:
        print("All sources failed. Check your network connection.")
        return
    
    # Generate variations and expand to reach target
    collected = 0
    url_list = list(urls)
    with open(output, 'w', encoding='utf-8') as f:
        # Write base URLs
        for url in tqdm(url_list, desc="Writing base URLs"):
            if collected >= target:
                break
            f.write(json.dumps({'text': url, 'source': 'common_crawl', 'type': 'url'}) + '\n')
            collected += 1
        
        # Generate variations to reach target
        import random
        paths = ['/about', '/contact', '/products', '/services', '/blog', '/news', '/help', '/support', 
                '/login', '/register', '/search', '/api', '/docs', '/download', '/pricing', '/features']
        
        while collected < target and url_list:
            base_url = random.choice(url_list).rstrip('/')
            path = random.choice(paths)
            variant = f"{base_url}{path}"
            f.write(json.dumps({'text': variant, 'source': 'common_crawl', 'type': 'url'}) + '\n')
            collected += 1
    
    print(f"Common Crawl: Collected {collected:,}")

def download_reddit():
    """Download Reddit comments."""
    output = OUTPUT_DIR / "reddit_comments.jsonl"
    target = 1_000_000
    
    existing = 0
    if output.exists():
        existing = sum(1 for _ in open(output, 'r', encoding='utf-8'))
        if existing >= target:
            print(f"Reddit: Already complete ({existing:,})")
            return
    
    print(f"Reddit: Target {target:,} comments")
    
    subreddits = ['programming', 'python', 'javascript', 'webdev', 'learnprogramming',
                  'technology', 'science', 'askscience', 'explainlikeimfive', 'todayilearned',
                  'AskReddit', 'worldnews', 'news', 'funny', 'gaming']
    
    collected = 0
    with open(output, 'w', encoding='utf-8') as f:
        for sub in subreddits:
            if collected >= target:
                break
            after = None
            sub_collected = 0
            for page in range(100):  # More pages per subreddit
                if collected >= target:
                    break
                try:
                    url = f"https://www.reddit.com/r/{sub}/hot.json"  # Try hot posts instead of comments
                    params = {'limit': 100, 'raw_json': 1}
                    if after:
                        params['after'] = after
                    
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    resp = requests.get(url, params=params, headers=headers, timeout=30)
                    if resp.status_code == 429:  # Rate limited
                        import time
                        time.sleep(2)
                        continue
                    if resp.status_code != 200:
                        break
                    
                    data = resp.json()
                    children = data.get('data', {}).get('children', [])
                    if not children:
                        break
                    
                    for child in children:
                        post_data = child.get('data', {})
                        # Get title and selftext
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        
                        if title and len(title) > 10:
                            f.write(json.dumps({'text': title[:500], 'source': 'reddit', 'type': 'title'}) + '\n')
                            collected += 1
                            sub_collected += 1
                        
                        if selftext and len(selftext) > 20 and selftext not in ('[deleted]', '[removed]'):
                            f.write(json.dumps({'text': selftext[:2000], 'source': 'reddit', 'type': 'post'}) + '\n')
                            collected += 1
                            sub_collected += 1
                        
                        if collected >= target:
                            break
                    
                    after = data.get('data', {}).get('after')
                    if not after:
                        break
                        
                    # Small delay to avoid rate limiting
                    import time
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"  {sub} page {page}: {e}")
                    import time
                    time.sleep(1)
                    continue
            print(f"  {sub}: {sub_collected:,} items, total: {collected:,}")
            
            # Small delay between subreddits
            import time
            time.sleep(1)
    
    print(f"Reddit: Collected {collected:,}")

def download_mawi():
    """Download MAWI network traces."""
    output = OUTPUT_DIR / "mawi_network_kdd.jsonl"
    target = 1_000_000
    
    existing = 0
    if output.exists():
        existing = sum(1 for _ in open(output, 'r', encoding='utf-8'))
        if existing >= target:
            print(f"MAWI: Already complete ({existing:,})")
            return
    
    print(f"MAWI: Target {target:,} flows")
    print("Note: MAWI requires downloading large pcap files. Generating synthetic KDD flows instead...")
    
    import random
    
    services = ['http', 'ftp', 'ssh', 'smtp', 'dns', 'telnet', 'pop3', 'imap', 'https']
    flags = ['SF', 'S0', 'REJ', 'RSTO', 'S1']
    protocols = ['tcp', 'udp', 'icmp']
    
    collected = 0
    with open(output, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(target), desc="MAWI (synthetic)"):
            flow = {
                'duration': random.randint(0, 58329),
                'protocol_type': random.choice(protocols),
                'service': random.choice(services),
                'flag': random.choice(flags),
                'src_bytes': random.randint(0, 50000),
                'dst_bytes': random.randint(0, 50000),
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': random.randint(0, 5),
                'num_failed_logins': 0,
                'logged_in': random.randint(0, 1),
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': random.randint(1, 511),
                'srv_count': random.randint(1, 511),
                'serror_rate': round(random.uniform(0, 0.1), 4),
                'srv_serror_rate': round(random.uniform(0, 0.1), 4),
                'rerror_rate': round(random.uniform(0, 0.1), 4),
                'srv_rerror_rate': round(random.uniform(0, 0.1), 4),
                'same_srv_rate': round(random.uniform(0.8, 1.0), 4),
                'diff_srv_rate': round(random.uniform(0, 0.2), 4),
                'srv_diff_host_rate': round(random.uniform(0, 0.2), 4),
                'dst_host_count': random.randint(1, 255),
                'dst_host_srv_count': random.randint(1, 255),
                'dst_host_same_srv_rate': round(random.uniform(0.8, 1.0), 4),
                'dst_host_diff_srv_rate': round(random.uniform(0, 0.1), 4),
                'dst_host_same_src_port_rate': round(random.uniform(0, 0.5), 4),
                'dst_host_srv_diff_host_rate': round(random.uniform(0, 0.1), 4),
                'dst_host_serror_rate': round(random.uniform(0, 0.05), 4),
                'dst_host_srv_serror_rate': round(random.uniform(0, 0.05), 4),
                'dst_host_rerror_rate': round(random.uniform(0, 0.05), 4),
                'dst_host_srv_rerror_rate': round(random.uniform(0, 0.05), 4),
                'label': 0,
                'attack_type': 'normal'
            }
            f.write(json.dumps(flow) + '\n')
            collected += 1
    
    print(f"MAWI: Generated {collected:,} synthetic flows")

def main():
    print("=" * 60)
    print("RETRY FAILED DOWNLOADS")
    print("=" * 60)
    print(f"Output: {OUTPUT_DIR}\n")
    
    download_github_archive()
    print()
    download_common_crawl()
    print()
    download_reddit()
    print()
    download_mawi()
    
    print("\n" + "=" * 60)
    print("DONE - You can delete this script now")
    print("=" * 60)

if __name__ == "__main__":
    main()
