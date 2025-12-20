"""Phase A2: Download real URL datasets (URLhaus + Tranco)."""
import os
import requests
from pathlib import Path


def download_file(url, dest_path, desc=""):
    """Download file with progress."""
    print(f"Downloading {desc or url}...")
    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  ✓ Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    base_path = Path(__file__).parent.parent
    url_dir = base_path / 'datasets' / 'url_analysis'
    url_dir.mkdir(parents=True, exist_ok=True)
    
    # URLhaus - recent malware URLs (no auth needed)
    download_file(
        "https://urlhaus.abuse.ch/downloads/csv_recent/",
        url_dir / "urlhaus.csv",
        "URLhaus malware URLs"
    )
    
    # Tranco Top 1M - benign domains (use latest list)
    download_file(
        "https://tranco-list.eu/top-1m.csv.zip",
        url_dir / "tranco_top1m.csv.zip",
        "Tranco Top 1M domains"
    )
    
    # Extract if zip downloaded
    import zipfile
    zip_path = url_dir / "tranco_top1m.csv.zip"
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(url_dir)
            print("  ✓ Extracted Tranco CSV")
        except: pass
    
    print("\n--- Parsing Downloaded Data ---")
    
    # Parse URLhaus
    malicious_urls = []
    urlhaus_path = url_dir / "urlhaus.csv"
    if urlhaus_path.exists():
        with open(urlhaus_path, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[2].startswith('"http'):
                    url = parts[2].strip('"')
                    malicious_urls.append(url)
        print(f"  Parsed {len(malicious_urls)} malicious URLs from URLhaus")
    
    # Parse Tranco
    benign_domains = []
    tranco_path = url_dir / "tranco_top1m.csv"
    if tranco_path.exists():
        with open(tranco_path, 'r', errors='ignore') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    benign_domains.append(parts[1])
        print(f"  Parsed {len(benign_domains)} benign domains from Tranco")
    
    # Save processed files
    if malicious_urls:
        with open(url_dir / "real_malicious_urls.txt", 'w') as f:
            f.write('\n'.join(malicious_urls[:100000]))
        print(f"  ✓ Saved {min(len(malicious_urls), 100000)} malicious URLs")
    
    if benign_domains:
        with open(url_dir / "real_benign_domains.txt", 'w') as f:
            f.write('\n'.join(benign_domains[:100000]))
        print(f"  ✓ Saved {min(len(benign_domains), 100000)} benign domains")
    
    print("\nDone! Run train_url.py to retrain with real data.")


if __name__ == "__main__":
    main()
