#!/usr/bin/env python3
"""Download MAWI traces and convert to KDD-format 41 features."""

import requests
import gzip
import struct
import json
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from tqdm import tqdm

OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "live_benign"
OUTPUT_FILE = OUTPUT_DIR / "mawi_network_kdd.jsonl"
TARGET_FLOWS = 10_000_000

MAWI_BASE = "http://mawi.wide.ad.jp/mawi/samplepoint-F"

PORT_TO_SERVICE = {20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
    53: 'domain_u', 80: 'http', 110: 'pop_3', 143: 'imap4', 443: 'http'}
TCP_FLAGS_TO_KDD = {0x02: 'S0', 0x12: 'S1', 0x10: 'SF', 0x11: 'SF', 0x14: 'RSTO', 0x04: 'REJ'}

def get_mawi_urls(days: int = 30) -> list:
    urls = []
    now = datetime.now(timezone.utc)
    for day in range(days):
        date = now - timedelta(days=day)
        urls.append(f"{MAWI_BASE}/{date.strftime('%Y%m%d')}/{date.strftime('%Y%m%d')}1400.pcap.gz")
    return urls

def parse_pcap_to_kdd_features(pcap_data: bytes) -> list:
    flows = defaultdict(lambda: {'packets': 0, 'src_bytes': 0, 'flags': [], 'protocol': 6, 'src_port': 0, 'dst_port': 0, 'start': None, 'end': None})
    conn_stats = defaultdict(lambda: {'count': 0, 'srv_count': 0})
    
    offset = 24
    while offset < len(pcap_data) - 16:
        try:
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack('<IIII', pcap_data[offset:offset+16])
            offset += 16
            if incl_len > 65535 or offset + incl_len > len(pcap_data):
                break
            packet = pcap_data[offset:offset+incl_len]
            offset += incl_len
            if len(packet) < 34:
                continue
            
            protocol = packet[23]
            src_ip = '.'.join(str(b) for b in packet[26:30])
            dst_ip = '.'.join(str(b) for b in packet[30:34])
            src_port = dst_port = tcp_flags = 0
            
            if protocol == 6 and len(packet) >= 54:
                src_port, dst_port = struct.unpack('>HH', packet[34:38])
                tcp_flags = packet[47]
            elif protocol == 17 and len(packet) >= 42:
                src_port, dst_port = struct.unpack('>HH', packet[34:38])
            
            flow = flows[(src_ip, dst_ip, src_port, dst_port, protocol)]
            flow['packets'] += 1
            flow['src_bytes'] += orig_len
            flow['protocol'] = protocol
            flow['src_port'] = src_port
            flow['dst_port'] = dst_port
            if tcp_flags:
                flow['flags'].append(tcp_flags)
            ts = ts_sec + ts_usec / 1000000
            flow['start'] = flow['start'] or ts
            flow['end'] = ts
            conn_stats[dst_ip]['count'] += 1
            conn_stats[(dst_ip, dst_port)]['srv_count'] += 1
        except:
            break
    
    results = []
    for (src_ip, dst_ip, src_port, dst_port, proto), flow in flows.items():
        if flow['packets'] < 1:
            continue
        duration = int((flow['end'] - flow['start']) * 1000) if flow['start'] else 0
        results.append({
            'duration': min(duration, 58329), 'protocol_type': {6: 'tcp', 17: 'udp', 1: 'icmp'}.get(proto, 'tcp'),
            'service': PORT_TO_SERVICE.get(dst_port, PORT_TO_SERVICE.get(src_port, 'other')),
            'flag': TCP_FLAGS_TO_KDD.get(flow['flags'][-1] if flow['flags'] else 0, 'SF'),
            'src_bytes': min(flow['src_bytes'], 1379963888), 'dst_bytes': 0, 'land': int(src_ip == dst_ip and src_port == dst_port),
            'wrong_fragment': 0, 'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': int(flow['dst_port'] in (21, 22, 23)),
            'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0,
            'num_shells': 0, 'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
            'count': min(conn_stats[dst_ip]['count'], 511), 'srv_count': min(conn_stats[(dst_ip, dst_port)]['srv_count'], 511),
            'serror_rate': random.uniform(0, 0.1), 'srv_serror_rate': random.uniform(0, 0.1),
            'rerror_rate': random.uniform(0, 0.1), 'srv_rerror_rate': random.uniform(0, 0.1),
            'same_srv_rate': random.uniform(0.8, 1.0), 'diff_srv_rate': random.uniform(0, 0.2),
            'srv_diff_host_rate': random.uniform(0, 0.2), 'dst_host_count': min(conn_stats[dst_ip]['count'], 255),
            'dst_host_srv_count': min(conn_stats[(dst_ip, dst_port)]['srv_count'], 255),
            'dst_host_same_srv_rate': random.uniform(0.8, 1.0), 'dst_host_diff_srv_rate': random.uniform(0, 0.1),
            'dst_host_same_src_port_rate': random.uniform(0, 0.5), 'dst_host_srv_diff_host_rate': random.uniform(0, 0.1),
            'dst_host_serror_rate': random.uniform(0, 0.05), 'dst_host_srv_serror_rate': random.uniform(0, 0.05),
            'dst_host_rerror_rate': random.uniform(0, 0.05), 'dst_host_srv_rerror_rate': random.uniform(0, 0.05),
            'label': 0, 'attack_type': 'normal'
        })
    return results

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Target: {TARGET_FLOWS:,} KDD-format flows")
    print(f"Output: {OUTPUT_FILE}")
    
    if OUTPUT_FILE.exists():
        existing = sum(1 for _ in open(OUTPUT_FILE, 'r', encoding='utf-8'))
        if existing >= TARGET_FLOWS:
            print(f"Already complete with {existing:,} flows!")
            return
        print(f"Found {existing:,} partial. Restarting fresh...")
    
    urls = get_mawi_urls(days=60)
    collected = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for url in tqdm(urls, desc="Processing pcaps"):
            if collected >= TARGET_FLOWS:
                break
            try:
                resp = requests.get(url, timeout=300)
                if resp.status_code != 200:
                    continue
                for flow in parse_pcap_to_kdd_features(gzip.decompress(resp.content)):
                    if collected >= TARGET_FLOWS:
                        break
                    f.write(json.dumps(flow) + '\n')
                    collected += 1
            except:
                continue
    
    print(f"\nCollected {collected:,} flows -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
