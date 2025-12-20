#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run shell command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Command failed: {e}")
        return False

def download_huggingface_phishing():
    """Download HuggingFace phishing dataset"""
    print("Downloading HuggingFace phishing dataset...")
    
    # Install datasets library if not present
    if not run_command("pip install datasets"):
        print("Failed to install datasets library")
        return False
    
    # Download dataset
    script = '''
from datasets import load_dataset
import pandas as pd

try:
    ds = load_dataset("ealvaradob/phishing-dataset", "url", trust_remote_code=True)
    df = ds["train"].to_pandas()
    df.to_csv("datasets/url_analysis/huggingface_phishing_urls.csv", index=False)
    print(f"Downloaded {len(df)} phishing URLs")
except Exception as e:
    print(f"Error downloading HuggingFace dataset: {e}")
'''
    
    with open("temp_download.py", "w") as f:
        f.write(script)
    
    success = run_command("python temp_download.py")
    os.remove("temp_download.py")
    return success

def download_payloads_all_the_things():
    """Download PayloadsAllTheThings repository"""
    print("Downloading PayloadsAllTheThings...")
    
    target_dir = "datasets/security_payloads/PayloadsAllTheThings"
    if os.path.exists(target_dir):
        print("PayloadsAllTheThings already exists, skipping...")
        return True
    
    return run_command(f"git clone --depth 1 https://github.com/swisskyrepo/PayloadsAllTheThings.git {target_dir}")

def download_seclists_fuzzing():
    """Download SecLists fuzzing payloads only"""
    print("Downloading SecLists (Fuzzing only)...")
    
    target_dir = "datasets/security_payloads/SecLists"
    if os.path.exists(target_dir):
        print("SecLists already exists, skipping...")
        return True
    
    # Clone with sparse checkout for fuzzing only
    commands = [
        f"git clone --depth 1 --filter=blob:none --sparse https://github.com/danielmiessler/SecLists.git {target_dir}",
        f"git sparse-checkout set Fuzzing"
    ]
    
    for cmd in commands:
        if not run_command(cmd, cwd=target_dir if "sparse-checkout" in cmd else None):
            return False
    
    return True

def main():
    """Download all missing datasets"""
    print("Downloading missing datasets...")
    
    # Ensure we're in the right directory (Windows path)
    project_dir = Path(__file__).parent.parent
    os.chdir(project_dir)
    
    # Create directories if they don't exist
    os.makedirs("datasets/url_analysis", exist_ok=True)
    os.makedirs("datasets/security_payloads", exist_ok=True)
    
    success_count = 0
    total_count = 3
    
    # Download datasets
    if download_huggingface_phishing():
        success_count += 1
        print("✅ HuggingFace phishing dataset downloaded")
    else:
        print("❌ Failed to download HuggingFace phishing dataset")
    
    if download_payloads_all_the_things():
        success_count += 1
        print("✅ PayloadsAllTheThings downloaded")
    else:
        print("❌ Failed to download PayloadsAllTheThings")
    
    if download_seclists_fuzzing():
        success_count += 1
        print("✅ SecLists (Fuzzing) downloaded")
    else:
        print("❌ Failed to download SecLists")
    
    print(f"\nCompleted: {success_count}/{total_count} datasets downloaded successfully")
    
    if success_count == total_count:
        print("🎉 All missing datasets downloaded!")
    else:
        print("⚠️  Some downloads failed. Check the errors above.")

if __name__ == "__main__":
    main()
