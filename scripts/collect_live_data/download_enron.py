#!/usr/bin/env python3
"""Download Enron email corpus."""

import tarfile
import json
import requests
import email
from email import policy
from pathlib import Path
from tqdm import tqdm
import re

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "enron_emails.jsonl"
TARGET_EMAILS = 500_000

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"

def clean_email_body(body: str) -> str:
    body = re.sub(r'-+\s*Forwarded.*?-+', '', body, flags=re.DOTALL)
    body = re.sub(r'-+\s*Original Message.*?-+', '', body, flags=re.DOTALL)
    body = re.sub(r'\n--\s*\n.*', '', body, flags=re.DOTALL)
    body = re.sub(r'\n{3,}', '\n\n', body)
    return body.strip()

def process_email_file(content: bytes) -> dict:
    try:
        msg = email.message_from_bytes(content, policy=policy.default)
        subject = msg.get('Subject', '') or ''
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_content()
                    break
        else:
            body = msg.get_content()
        if isinstance(body, bytes):
            body = body.decode('utf-8', errors='ignore')
        return {'subject': subject[:200], 'body': clean_email_body(body)[:5000]}
    except:
        return None

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_EMAILS:,} emails")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE, 'r', encoding='utf-8'))
        if existing >= TARGET_EMAILS:
            print(f"Already complete with {existing:,} emails!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")
    
    tar_path = OUTPUT_DIR / "enron_mail.tar.gz"
    
    if not tar_path.exists():
        print(f"Downloading Enron corpus (~400MB)...")
        try:
            resp = requests.get(ENRON_URL, stream=True, timeout=600)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            with open(tar_path, 'wb') as f:
                with tqdm(total=total, unit='B', unit_scale=True) as pbar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except Exception as e:
            print(f"Download failed: {e}")
            return
    
    print("Extracting emails...")
    collected = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            
            for member in tqdm(members, desc="Processing"):
                if collected >= TARGET_EMAILS:
                    break
                try:
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    email_data = process_email_file(f.read())
                    if email_data:
                        if email_data['subject']:
                            out.write(json.dumps({'text': email_data['subject'], 'source': 'enron', 'type': 'email_subject'}) + '\n')
                            collected += 1
                        if email_data['body'] and len(email_data['body']) > 50:
                            out.write(json.dumps({'text': email_data['body'], 'source': 'enron', 'type': 'email_body'}) + '\n')
                            collected += 1
                except:
                    continue
    
    print(f"\nCollected {collected:,} emails -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
