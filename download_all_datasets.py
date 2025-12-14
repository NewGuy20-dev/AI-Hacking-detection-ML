import os
import requests
import pandas as pd
from urllib.parse import urlparse
import zipfile
import gzip
import tarfile

class CyberDatasetDownloader:
    def __init__(self, base_dir="datasets"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def download_file(self, url, filename):
        """Download file with progress"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✓ Downloaded: {filename}")
            return True
        except Exception as e:
            print(f"✗ Failed to download {url}: {e}")
            return False
    
    def download_nsl_kdd(self):
        """NSL-KDD Dataset"""
        print("Downloading NSL-KDD Dataset...")
        os.makedirs(f"{self.base_dir}/nsl_kdd", exist_ok=True)
        
        urls = {
            'train': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
            'test': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
        }
        
        for name, url in urls.items():
            self.download_file(url, f"{self.base_dir}/nsl_kdd/{name}.csv")
    
    def download_cicids2017(self):
        """CICIDS2017 Dataset"""
        print("Downloading CICIDS2017 Dataset...")
        os.makedirs(f"{self.base_dir}/cicids2017", exist_ok=True)
        
        # Note: Full dataset requires registration, using sample
        sample_url = "https://www.unb.ca/cic/datasets/ids-2017.html"
        print(f"CICIDS2017 requires registration at: {sample_url}")
    
    def download_unsw_nb15(self):
        """UNSW-NB15 Dataset"""
        print("Downloading UNSW-NB15 Dataset...")
        os.makedirs(f"{self.base_dir}/unsw_nb15", exist_ok=True)
        
        urls = [
            "https://cloudstor.aarnet.edu.au/plus/s/umT99TnxvbpkkoE/download",
            "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download"
        ]
        
        for i, url in enumerate(urls):
            self.download_file(url, f"{self.base_dir}/unsw_nb15/part_{i+1}.csv")
    
    def download_kdd99(self):
        """KDD Cup 1999 Dataset"""
        print("Downloading KDD Cup 1999 Dataset...")
        os.makedirs(f"{self.base_dir}/kdd99", exist_ok=True)
        
        urls = {
            'train': 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data.gz',
            'test': 'http://kdd.ics.uci.edu/databases/kddcup99/corrected.gz'
        }
        
        for name, url in urls.items():
            filename = f"{self.base_dir}/kdd99/{name}.gz"
            if self.download_file(url, filename):
                # Extract gz file
                with gzip.open(filename, 'rb') as f_in:
                    with open(f"{self.base_dir}/kdd99/{name}.csv", 'wb') as f_out:
                        f_out.write(f_in.read())
    
    def download_malware_urls(self):
        """Malicious URLs Dataset"""
        print("Downloading Malicious URLs Dataset...")
        os.makedirs(f"{self.base_dir}/malware_urls", exist_ok=True)
        
        # URLhaus API
        url = "https://urlhaus-api.abuse.ch/v1/urls/recent/"
        try:
            response = requests.post(url)
            with open(f"{self.base_dir}/malware_urls/urlhaus_recent.json", 'w') as f:
                f.write(response.text)
            print("✓ Downloaded URLhaus recent URLs")
        except Exception as e:
            print(f"✗ Failed to download URLhaus: {e}")
    
    def download_phishing_dataset(self):
        """Phishing Websites Dataset"""
        print("Downloading Phishing Dataset...")
        os.makedirs(f"{self.base_dir}/phishing", exist_ok=True)
        
        # UCI Phishing dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
        self.download_file(url, f"{self.base_dir}/phishing/phishing_dataset.arff")
    
    def download_credit_fraud(self):
        """Credit Card Fraud Dataset"""
        print("Credit Card Fraud Dataset requires Kaggle API...")
        print("Visit: https://www.kaggle.com/mlg-ulb/creditcardfraud")
    
    def download_spam_dataset(self):
        """Spam Email Dataset"""
        print("Downloading Spam Dataset...")
        os.makedirs(f"{self.base_dir}/spam", exist_ok=True)
        
        # SpamAssassin public corpus
        url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"
        filename = f"{self.base_dir}/spam/spam_corpus.tar.bz2"
        
        if self.download_file(url, filename):
            # Extract tar.bz2
            with tarfile.open(filename, 'r:bz2') as tar:
                tar.extractall(f"{self.base_dir}/spam/")
    
    def download_alexa_top_domains(self):
        """Alexa Top 1M Domains"""
        print("Downloading Alexa Top Domains...")
        os.makedirs(f"{self.base_dir}/domains", exist_ok=True)
        
        url = "http://s3.amazonaws.com/alexa-static/top-1m.csv.zip"
        filename = f"{self.base_dir}/domains/alexa_top_1m.zip"
        
        if self.download_file(url, filename):
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(f"{self.base_dir}/domains/")
    
    def download_all(self):
        """Download all available datasets"""
        print("Starting comprehensive cybersecurity dataset download...")
        
        self.download_nsl_kdd()
        self.download_kdd99()
        self.download_malware_urls()
        self.download_phishing_dataset()
        self.download_spam_dataset()
        self.download_alexa_top_domains()
        
        # Datasets requiring registration
        print("\n--- Datasets requiring manual download/registration ---")
        self.download_cicids2017()
        self.download_unsw_nb15()
        self.download_credit_fraud()
        
        print("\n✓ Dataset download complete!")
        print(f"All datasets saved to: {self.base_dir}/")

if __name__ == "__main__":
    downloader = CyberDatasetDownloader()
    downloader.download_all()
