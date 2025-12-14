import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import os

class DatasetPreprocessor:
    def __init__(self, datasets_dir="datasets"):
        self.datasets_dir = datasets_dir
        
    def preprocess_nsl_kdd(self):
        """Preprocess NSL-KDD dataset"""
        columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                   'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
                   'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
                   'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
                   'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                   'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
                   'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                   'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                   'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                   'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty']
        
        try:
            train_df = pd.read_csv(f'{self.datasets_dir}/nsl_kdd/train.csv', names=columns)
            test_df = pd.read_csv(f'{self.datasets_dir}/nsl_kdd/test.csv', names=columns)
            
            # Create binary labels
            train_df['is_attack'] = (train_df['attack_type'] != 'normal').astype(int)
            test_df['is_attack'] = (test_df['attack_type'] != 'normal').astype(int)
            
            return train_df, test_df
        except FileNotFoundError:
            print("NSL-KDD dataset not found")
            return None, None
    
    def preprocess_phishing(self):
        """Preprocess phishing dataset"""
        try:
            # Read ARFF file (simplified)
            with open(f'{self.datasets_dir}/phishing/phishing_dataset.arff', 'r') as f:
                lines = f.readlines()
            
            # Find data section
            data_start = None
            for i, line in enumerate(lines):
                if line.strip().lower() == '@data':
                    data_start = i + 1
                    break
            
            if data_start:
                data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
                data = [line.split(',') for line in data_lines]
                
                # Create DataFrame
                columns = [f'feature_{i}' for i in range(len(data[0])-1)] + ['class']
                df = pd.DataFrame(data, columns=columns)
                
                # Convert to numeric
                for col in df.columns[:-1]:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['is_phishing'] = (df['class'] == '1').astype(int)
                return df
        except FileNotFoundError:
            print("Phishing dataset not found")
            return None
    
    def preprocess_malware_urls(self):
        """Preprocess malware URLs dataset"""
        try:
            with open(f'{self.datasets_dir}/malware_urls/urlhaus_recent.json', 'r') as f:
                data = json.load(f)
            
            if 'urls' in data:
                df = pd.DataFrame(data['urls'])
                df['is_malware'] = 1  # All URLs from URLhaus are malicious
                return df
        except FileNotFoundError:
            print("Malware URLs dataset not found")
            return None
    
    def preprocess_alexa_domains(self):
        """Preprocess Alexa top domains"""
        try:
            df = pd.read_csv(f'{self.datasets_dir}/domains/top-1m.csv', 
                           names=['rank', 'domain'])
            df['is_malware'] = 0  # Alexa domains are legitimate
            return df
        except FileNotFoundError:
            print("Alexa domains dataset not found")
            return None
    
    def create_unified_dataset(self):
        """Create unified dataset from all sources"""
        print("Creating unified cybersecurity dataset...")
        
        datasets = {}
        
        # Network traffic datasets
        train_nsl, test_nsl = self.preprocess_nsl_kdd()
        if train_nsl is not None:
            datasets['nsl_kdd_train'] = train_nsl
            datasets['nsl_kdd_test'] = test_nsl
        
        # Phishing dataset
        phishing_df = self.preprocess_phishing()
        if phishing_df is not None:
            datasets['phishing'] = phishing_df
        
        # URL datasets
        malware_urls = self.preprocess_malware_urls()
        if malware_urls is not None:
            datasets['malware_urls'] = malware_urls
        
        alexa_domains = self.preprocess_alexa_domains()
        if alexa_domains is not None:
            datasets['alexa_domains'] = alexa_domains
        
        return datasets
    
    def generate_summary(self, datasets):
        """Generate dataset summary"""
        print("\n=== CYBERSECURITY DATASETS SUMMARY ===")
        
        for name, df in datasets.items():
            print(f"\n{name.upper()}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)[:5]}...")
            
            # Attack distribution if available
            if 'is_attack' in df.columns:
                attack_dist = df['is_attack'].value_counts()
                print(f"  Attack distribution: Normal={attack_dist.get(0, 0)}, Attack={attack_dist.get(1, 0)}")
            elif 'is_phishing' in df.columns:
                phish_dist = df['is_phishing'].value_counts()
                print(f"  Phishing distribution: Legitimate={phish_dist.get(0, 0)}, Phishing={phish_dist.get(1, 0)}")
            elif 'is_malware' in df.columns:
                malware_dist = df['is_malware'].value_counts()
                print(f"  Malware distribution: Clean={malware_dist.get(0, 0)}, Malware={malware_dist.get(1, 0)}")

if __name__ == "__main__":
    preprocessor = DatasetPreprocessor()
    datasets = preprocessor.create_unified_dataset()
    preprocessor.generate_summary(datasets)
