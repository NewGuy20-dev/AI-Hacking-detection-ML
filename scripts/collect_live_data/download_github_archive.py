#!/usr/bin/env python3
"""Download code snippets from GitHub Archive (NO repo cloning)."""

import gzip
import json
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "github_snippets.jsonl"
TARGET_SNIPPETS = 100_000_000

GH_ARCHIVE_BASE = "https://data.gharchive.org"

def get_archive_urls(days: int = 30) -> list:
    urls = []
    now = datetime.now(timezone.utc)
    for day in range(days):
        date = now - timedelta(days=day)
        for hour in range(24):
            urls.append(f"{GH_ARCHIVE_BASE}/{date.strftime('%Y-%m-%d')}-{hour}.json.gz")
    return urls

def extract_text_from_event(event: dict) -> list:
    texts = []
    event_type = event.get('type', '')
    payload = event.get('payload', {})
    
    if event_type == 'PushEvent':
        for commit in payload.get('commits', []):
            msg = commit.get('message', '')
            if msg and len(msg) > 10:
                texts.append(('commit_message', msg[:500]))
    elif event_type in ('IssuesEvent', 'PullRequestEvent'):
        item = payload.get('issue') or payload.get('pull_request', {})
        if item.get('title'):
            texts.append(('issue_title', item['title'][:200]))
        if item.get('body') and len(item['body']) > 20:
            texts.append(('issue_body', item['body'][:1000]))
    elif event_type in ('IssueCommentEvent', 'CommitCommentEvent', 'PullRequestReviewCommentEvent'):
        body = payload.get('comment', {}).get('body', '')
        if body and len(body) > 20:
            texts.append(('comment', body[:500]))
    return texts

def download_and_process_archive(url: str, max_events: int = 50000) -> list:
    snippets = []
    try:
        resp = requests.get(url, timeout=120, stream=True)
        if resp.status_code != 200:
            return []
        content = gzip.decompress(resp.content).decode('utf-8', errors='ignore')
        for line in content.split('\n')[:max_events]:
            if not line.strip():
                continue
            try:
                snippets.extend(extract_text_from_event(json.loads(line)))
            except:
                pass
            if len(snippets) >= max_events:
                break
    except:
        pass
    return snippets

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GitHub Archive Downloader (NO repo cloning)")
    print("=" * 60)
    print(f"\nTarget: {TARGET_SNIPPETS:,} snippets")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE, 'r', encoding='utf-8'))
        if existing >= TARGET_SNIPPETS:
            print(f"Already complete with {existing:,} snippets!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")
    
    urls = get_archive_urls(days=60)
    print(f"Archive files: {len(urls)}")
    
    collected = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(download_and_process_archive, url): url for url in urls}
            
            with tqdm(total=TARGET_SNIPPETS, desc="Collecting") as pbar:
                for future in as_completed(futures):
                    if collected >= TARGET_SNIPPETS:
                        break
                    for snippet_type, text in future.result():
                        if collected >= TARGET_SNIPPETS:
                            break
                        f.write(json.dumps({'text': text, 'source': 'github_archive', 'type': snippet_type}) + '\n')
                        collected += 1
                        pbar.update(1)
    
    print(f"\nCollected {collected:,} snippets -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
