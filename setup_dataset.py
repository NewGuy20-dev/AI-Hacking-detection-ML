import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

def download_nsl_kdd():
    """Download NSL-KDD dataset"""
    os.makedirs('data', exist_ok=True)
    
    urls = {
        'train': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt',
        'test': 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
    }
    
    for name, url in urls.items():
        print(f"Downloading {name} data...")
        response = requests.get(url)
        with open(f'data/nsl_kdd_{name}.csv', 'wb') as f:
            f.write(response.content)

def load_and_preprocess():
    """Load and basic preprocessing"""
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
    
    train_df = pd.read_csv('data/nsl_kdd_train.csv', names=columns)
    test_df = pd.read_csv('data/nsl_kdd_test.csv', names=columns)
    
    # Binary classification: normal vs attack
    train_df['is_attack'] = (train_df['attack_type'] != 'normal').astype(int)
    test_df['is_attack'] = (test_df['attack_type'] != 'normal').astype(int)
    
    return train_df, test_df

if __name__ == "__main__":
    download_nsl_kdd()
    train_df, test_df = load_and_preprocess()
    print(f"Training data: {train_df.shape}")
    print(f"Test data: {test_df.shape}")
    print(f"Attack distribution: {train_df['is_attack'].value_counts()}")
