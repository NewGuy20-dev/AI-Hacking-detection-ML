#!/usr/bin/env python3
"""Download Reddit comments from API."""

import json
import requests
from pathlib import Path
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "reddit_comments.jsonl"
TARGET_COMMENTS = 1_000_000

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_COMMENTS:,} comments")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE, 'r', encoding='utf-8'))
        if existing >= TARGET_COMMENTS:
            print(f"Already complete with {existing:,} comments!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")
    
    # Use Reddit JSON API (no auth needed for public)
    subreddits = ['programming', 'python', 'javascript', 'webdev', 'learnprogramming', 
                  'technology', 'science', 'askscience', 'explainlikeimfive', 'todayilearned']
    
    collected = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        with tqdm(total=TARGET_COMMENTS, desc="Collecting") as pbar:
            for subreddit in subreddits:
                if collected >= TARGET_COMMENTS:
                    break
                
                after = None
                for _ in range(100):  # Max 100 pages per subreddit
                    if collected >= TARGET_COMMENTS:
                        break
                    try:
                        url = f"https://www.reddit.com/r/{subreddit}/comments.json"
                        params = {'limit': 100, 'raw_json': 1}
                        if after:
                            params['after'] = after
                        
                        resp = requests.get(url, params=params, headers={'User-Agent': 'DataCollector/1.0'}, timeout=30)
                        if resp.status_code != 200:
                            break
                        
                        data = resp.json()
                        children = data.get('data', {}).get('children', [])
                        if not children:
                            break
                        
                        for child in children:
                            body = child.get('data', {}).get('body', '')
                            if body and len(body) > 20 and body not in ('[deleted]', '[removed]'):
                                f.write(json.dumps({
                                    'text': body[:2000],
                                    'source': 'reddit',
                                    'type': 'comment',
                                    'subreddit': subreddit
                                }) + '\n')
                                collected += 1
                                pbar.update(1)
                        
                        after = data.get('data', {}).get('after')
                        if not after:
                            break
                    except:
                        break
    
    print(f"\nCollected {collected:,} comments -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
