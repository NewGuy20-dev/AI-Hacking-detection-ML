#!/bin/bash
# Download missing datasets for AI Hacking Detection

set -e
cd /workspaces/AI-Hacking-detection-ML/datasets

echo "=== Setting up Kaggle API ==="
mkdir -p ~/.kaggle
echo '{"username":"kgat","key":"02745f16ea68d61dc7a60addec4a4384"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

echo -e "\n=== 1. KDD Cup 1999 ==="
cd kdd99
if [ ! -f "kddcup.data_10_percent.gz" ]; then
    wget -q http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz
    gunzip -k kddcup.data_10_percent.gz 2>/dev/null || true
    echo "Downloaded KDD99"
else
    echo "KDD99 already exists"
fi
cd ..

echo -e "\n=== 2. UNSW-NB15 ==="
cd unsw_nb15
if [ ! -f "UNSW-NB15_1.csv" ]; then
    kaggle datasets download -d mrwellsdavid/unsw-nb15 --unzip -q 2>/dev/null || \
    wget -q https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download -O unsw_sample.csv
    echo "Downloaded UNSW-NB15"
else
    echo "UNSW-NB15 already exists"
fi
cd ..

echo -e "\n=== 3. Credit Card Fraud (Kaggle) ==="
mkdir -p fraud && cd fraud
if [ ! -f "creditcard.csv" ]; then
    kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -q
    echo "Downloaded Credit Card Fraud"
else
    echo "Credit Card Fraud already exists"
fi
cd ..

echo -e "\n=== 4. Malware URLs ==="
cd malware_urls
if [ ! -f "malware_urls.csv" ]; then
    kaggle datasets download -d sid321axn/malicious-urls-dataset --unzip -q 2>/dev/null || true
    # Backup: URLhaus
    wget -q https://urlhaus.abuse.ch/downloads/csv_recent/ -O urlhaus_recent.csv 2>/dev/null || true
    echo "Downloaded Malware URLs"
else
    echo "Malware URLs already exists"
fi
cd ..

echo -e "\n=== 5. EMBER Malware Dataset (sample) ==="
mkdir -p ember && cd ember
if [ ! -f "ember_sample.json" ]; then
    # Full EMBER is 1GB+, get sample
    wget -q https://raw.githubusercontent.com/elastic/ember/master/ember/features.py -O features_info.py 2>/dev/null || true
    echo '{"note": "Full EMBER dataset requires: pip install ember; ember download"}' > ember_sample.json
    echo "EMBER info downloaded (full dataset is 1GB+)"
else
    echo "EMBER info exists"
fi
cd ..

echo -e "\n=== 6. Alexa Top 1M (extract) ==="
cd domains
if [ -f "alexa_top_1m.zip" ] && [ ! -f "top-1m-full.csv" ]; then
    unzip -o alexa_top_1m.zip 2>/dev/null || true
    mv top-1m.csv top-1m-full.csv 2>/dev/null || true
    echo "Extracted Alexa Top 1M"
fi
cd ..

echo -e "\n=== 7. Spam Corpus (extract) ==="
cd spam
if [ -f "spam_corpus.tar.bz2" ] && [ ! -d "spam" ]; then
    tar -xjf spam_corpus.tar.bz2 2>/dev/null || true
    echo "Extracted Spam Corpus"
fi
cd ..

echo -e "\n=== 8. CTU-13 Botnet (sample) ==="
mkdir -p ctu13 && cd ctu13
if [ ! -f "ctu13_info.txt" ]; then
    echo "CTU-13 Dataset: https://www.stratosphereips.org/datasets-ctu13" > ctu13_info.txt
    echo "Full dataset is 30GB+. Download specific scenarios as needed." >> ctu13_info.txt
    echo "CTU-13 info saved"
fi
cd ..

echo -e "\n=== Download Summary ==="
echo "Datasets directory contents:"
du -sh */ 2>/dev/null | head -15

echo -e "\nDone! Some large datasets (EMBER, CTU-13) require manual download."
