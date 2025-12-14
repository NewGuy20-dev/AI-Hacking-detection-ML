#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to download and setup a repository
download_repo() {
    local url=$1
    local target_dir=$2
    local zip_name=$3
    
    echo -e "${YELLOW}Downloading ${target_dir}...${NC}"
    wget -q "$url" -O "$zip_name"
    unzip -q "$zip_name"
    rm "$zip_name"
    
    # Move/rename based on the extracted directory name
    if [ "$4" != "" ]; then
        mv "$4" "$target_dir"
    fi
    echo -e "${GREEN}✓ ${target_dir} downloaded${NC}"
}

# Array to store missing repositories
declare -a repos=(
    "fuzzdb|https://github.com/fuzzdb-project/fuzzdb/archive/master.zip|fuzzdb.zip|fuzzdb-master"
    "danielmiessler|https://github.com/danielmiessler/SecLists/archive/master.zip|seclists.zip|SecLists-master"
    "xsuperbug|https://github.com/xsuperbug/payloads/archive/master.zip|xsuperbug.zip|payloads-master"
    "NickSanzotta|https://github.com/NickSanzotta/BurpIntruder/archive/master.zip|nicksanzotta.zip|BurpIntruder-master"
    "7ioSecurity|https://github.com/7ioSecurity/XSS-Payloads/archive/master.zip|7iosecurity.zip|XSS-Payloads-master"
    "shadsidd|https://github.com/shadsidd/Automated-XSS-Finder/archive/master.zip|shadsidd.zip|Automated-XSS-Finder-master"
    "tennc|https://gist.github.com/tennc/4026cfd0925aaad0a655/archive/c5741a19a0007bac49caf6cfccc93b296ec38cf0.zip|tennc.zip|4026cfd0925aaad0a655-c5741a19a0007bac49caf6cfccc93b296ec38cf0"
    "sqlifuzzer|https://github.com/ContactLeft/sqlifuzzer/archive/master.zip|sqlifuzzer.zip|sqlifuzzer-master"
    "wfuzz|https://github.com/xmendez/wfuzz/archive/master.zip|wfuzz.zip|wfuzz-master"
    "big-list-of-naughty-strings|https://github.com/minimaxir/big-list-of-naughty-strings/archive/master.zip|blns.zip|big-list-of-naughty-strings-master"
    "Commodity-Injection-Signatures|https://github.com/xsscx/Commodity-Injection-Signatures/archive/master.zip|xsscx.zip|Commodity-Injection-Signatures-master"
    "subbrute|https://github.com/TheRook/subbrute/archive/master.zip|subbrute.zip|subbrute-master"
    "RobotsDisallowed|https://github.com/danielmiessler/RobotsDisallowed/archive/master.zip|robots.zip|RobotsDisallowed-master"
    "HashCollision-DOS-POC|https://github.com/FireFart/HashCollision-DOS-POC/archive/master.zip|hashcollision.zip|HashCollision-DOS-POC-master"
    "aspHashDoS|https://github.com/FireFart/aspHashDoS/archive/master.zip|asphashdos.zip|aspHashDoS-master"
    "PayloadsAllTheThings|https://github.com/swisskyrepo/PayloadsAllTheThings/archive/master.zip|patt.zip|PayloadsAllTheThings-master"
    "IntruderPayloads|https://github.com/1N3/IntruderPayloads/archive/master.zip|intruder.zip|IntruderPayloads-master"
    "Open-Redirect-Payloads|https://github.com/cujanovic/Open-Redirect-Payloads/archive/master.zip|openredir.zip|Open-Redirect-Payloads-master"
    "Content-Bruteforcing-Wordlist|https://github.com/cujanovic/Content-Bruteforcing-Wordlist/archive/master.zip|contentbrute.zip|Content-Bruteforcing-Wordlist-master"
    "subdomain-bruteforce-list|https://github.com/cujanovic/subdomain-bruteforce-list/archive/master.zip|subdomainbrute.zip|subdomain-bruteforce-list-master"
    "CRLF-Injection-Payloads|https://github.com/cujanovic/CRLF-Injection-Payloads/archive/master.zip|crlf.zip|CRLF-Injection-Payloads-master"
    "Virtual-host-wordlist|https://github.com/cujanovic/Virtual-host-wordlist/archive/master.zip|vhost.zip|Virtual-host-wordlist-master"
    "dirsearch-wordlist|https://github.com/cujanovic/dirsearch-wordlist/archive/master.zip|dirsearch.zip|dirsearch-wordlist-master"
    "password-lists|https://github.com/lavalamp-/password-lists/archive/master.zip|passwords.zip|password-lists-master"
    "ics-default-passwords|https://github.com/arnaudsoullie/ics-default-passwords/archive/master.zip|ics.zip|ics-default-passwords-master"
    "SCADAPASS|https://github.com/scadastrangelove/SCADAPASS/archive/master.zip|scadapass.zip|SCADAPASS-master"
    "wordlist|https://github.com/jeanphorn/wordlist/archive/master.zip|jeanphorn.zip|wordlist-master"
    "PassList|https://github.com/j3ers3/PassList/archive/master.zip|passlist.zip|PassList-master"
    "awesome-default-passwords|https://github.com/nyxxxie/awesome-default-passwords/archive/master.zip|awesomepass.zip|awesome-default-passwords-master"
    "web-cve-tests|https://github.com/foospidy/web-cve-tests/archive/master.zip|webcve.zip|web-cve-tests-master"
    "Tiny-XSS-Payloads|https://github.com/terjanq/Tiny-XSS-Payloads/archive/master.zip|tinyxss.zip|Tiny-XSS-Payloads-master"
)

# Check which repositories are missing
missing_repos=()
existing_count=0

echo -e "${YELLOW}Checking existing repositories...${NC}"
for repo in "${repos[@]}"; do
    IFS='|' read -r dir_name url zip_name extracted_name <<< "$repo"
    if [ -d "$dir_name" ]; then
        echo -e "${GREEN}✓ ${dir_name} already exists${NC}"
        ((existing_count++))
    else
        echo -e "${RED}✗ ${dir_name} missing${NC}"
        missing_repos+=("$repo")
    fi
done

echo ""
echo -e "${YELLOW}Summary: ${existing_count}/${#repos[@]} repositories found${NC}"
echo ""

# Download missing repositories
if [ ${#missing_repos[@]} -eq 0 ]; then
    echo -e "${GREEN}All repositories are already downloaded!${NC}"
else
    echo -e "${YELLOW}Downloading ${#missing_repos[@]} missing repositories...${NC}"
    echo ""
    
    for repo in "${missing_repos[@]}"; do
        IFS='|' read -r dir_name url zip_name extracted_name <<< "$repo"
        download_repo "$url" "$dir_name" "$zip_name" "$extracted_name"
    done
    
    echo ""
    echo -e "${GREEN}Download complete!${NC}"
fi

# Decompress payload files if they exist and are still compressed
echo ""
echo -e "${YELLOW}Checking for compressed payload files...${NC}"

compressed_files=(
    "ctf/maccdc2010.txt.gz"
    "ctf/maccdc2011.txt.gz"
    "ctf/maccdc2012.txt.gz"
    "ctf/ists12_2015.txt.gz"
    "ctf/defcon20.txt.gz"
)

for file in "${compressed_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${YELLOW}Extracting ${file}...${NC}"
        gunzip "$file"
        echo -e "${GREEN}✓ ${file} extracted${NC}"
    elif [ -f "${file%.gz}" ]; then
        echo -e "${GREEN}✓ ${file%.gz} already extracted${NC}"
    fi
done

echo ""
echo -e "${GREEN}All done!${NC}"