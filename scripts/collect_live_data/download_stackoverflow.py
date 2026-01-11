#!/usr/bin/env python3
"""Download Stack Overflow posts from API with authentication."""

import json
import requests
import re
from pathlib import Path
from tqdm import tqdm
import time

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "stackoverflow_posts.jsonl"
TARGET_POSTS = 1_000_000

API_KEY = "rl_iYiqu1W6UND1pBi4uT334TDSx"

def clean_html(text: str) -> str:
    text = re.sub(r'<code>.*?</code>', '[CODE]', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_POSTS:,} posts")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Using API key for 10k daily quota")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE, 'r', encoding='utf-8'))
        if existing >= TARGET_POSTS:
            print(f"Already complete with {existing:,} posts!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")
    
    collected = 0
    page = 1
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        with tqdm(total=TARGET_POSTS, desc="Collecting") as pbar:
            while collected < TARGET_POSTS:
                try:
                    resp = requests.get(
                        "https://api.stackexchange.com/2.3/questions",
                        params={
                            'key': API_KEY,
                            'page': page,
                            'pagesize': 100,
                            'order': 'desc',
                            'sort': 'activity',
                            'site': 'stackoverflow',
                            'filter': '!nNPvSNdWme'  # Include body
                        },
                        timeout=30
                    )
                    
                    if resp.status_code == 400:
                        print(f"\nAPI error 400 - trying different filter...")
                        resp = requests.get(
                            "https://api.stackexchange.com/2.3/questions",
                            params={
                                'key': API_KEY,
                                'page': page,
                                'pagesize': 100,
                                'order': 'desc',
                                'sort': 'activity',
                                'site': 'stackoverflow',
                            },
                            timeout=30
                        )
                    
                    if resp.status_code != 200:
                        print(f"\nAPI error: {resp.status_code}")
                        break
                    
                    data = resp.json()
                    
                    if 'items' not in data or not data['items']:
                        print("\nNo more items")
                        break
                    
                    for item in data['items']:
                        title = item.get('title', '')
                        body = clean_html(item.get('body', '')) if 'body' in item else ''
                        tags = ' '.join(item.get('tags', []))
                        
                        if title:
                            f.write(json.dumps({'text': title, 'source': 'stackoverflow', 'type': 'question_title'}) + '\n')
                            collected += 1
                            pbar.update(1)
                        
                        if body and len(body) > 50:
                            f.write(json.dumps({'text': body[:2000], 'source': 'stackoverflow', 'type': 'question_body'}) + '\n')
                            collected += 1
                            pbar.update(1)
                        
                        if tags:
                            f.write(json.dumps({'text': tags, 'source': 'stackoverflow', 'type': 'tags'}) + '\n')
                            collected += 1
                            pbar.update(1)
                        
                        if collected >= TARGET_POSTS:
                            break
                    
                    page += 1
                    
                    # Check quota
                    quota = data.get('quota_remaining', 0)
                    if quota < 100:
                        print(f"\nQuota low: {quota} remaining")
                        break
                    
                    if not data.get('has_more', False):
                        print("\nNo more pages")
                        break
                    
                    # Small delay to be nice to API
                    if page % 100 == 0:
                        time.sleep(1)
                        
                except requests.exceptions.RequestException as e:
                    print(f"\nRequest error: {e}")
                    time.sleep(5)
                    continue
    
    print(f"\nCollected {collected:,} posts -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
