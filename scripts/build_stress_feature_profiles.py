#!/usr/bin/env python3
"""Build feature quantile profiles for stress-test generators."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import random
import numpy as np

NET_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]


def iter_jsonl(path):
    with Path(path).open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def sample_rows(path, max_samples, feature_order, category_fn, max_lines=None):
    per_category = defaultdict(list)
    count = 0
    for obj in iter_jsonl(path):
        if max_lines is not None and count >= max_lines:
            break
        category = category_fn(obj)
        if category is None:
            continue
        rows = per_category[category]
        if len(rows) < max_samples:
            rows.append([obj.get(k, 0.0) for k in feature_order])
        count += 1
    return per_category, count


def build_profiles(rows_by_category, feature_order):
    profiles = {}
    for category, rows in rows_by_category.items():
        if not rows:
            continue
        data = np.asarray(rows, dtype=np.float64)
        p01 = np.nanpercentile(data, 1, axis=0)
        p50 = np.nanpercentile(data, 50, axis=0)
        p99 = np.nanpercentile(data, 99, axis=0)
        profiles[category] = {
            'p01': p01.tolist(),
            'p50': p50.tolist(),
            'p99': p99.tolist(),
        }
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=20000, help='Max samples per category per dataset')
    parser.add_argument('--max-lines', type=int, default=500000, help='Max lines to scan per dataset file')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    base = Path('datasets')
    out_dir = Path('configs') / 'stress_test' / 'feature_profiles'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Host profiles
    host_paths = [
        base / 'live_benign' / 'host_behavior_benign.jsonl',
        base / 'host_behavior' / 'synthetic_500k.jsonl',
    ]

    host_feature_order = None
    for p in host_paths:
        if p.exists():
            for obj in iter_jsonl(p):
                host_feature_order = [k for k in obj.keys() if k not in ('label', 'category')]
                break
        if host_feature_order:
            break

    if host_feature_order:
        host_rows = defaultdict(list)
        for p in host_paths:
            if not p.exists():
                continue
            def host_category(obj):
                label = obj.get('label', 0)
                if int(label) == 0:
                    return 'normal'
                cat = str(obj.get('category', '')).strip().lower()
                return cat if cat else None
            rows, _ = sample_rows(p, args.max_samples, host_feature_order, host_category, max_lines=args.max_lines)
            for cat, data in rows.items():
                host_rows[cat].extend(data)
        host_profiles = build_profiles(host_rows, host_feature_order)
        host_out = {
            'model': 'host',
            'features': host_feature_order,
            'profiles': host_profiles,
        }
        with open(out_dir / 'host_profile_v1.json', 'w', encoding='utf-8') as f:
            json.dump(host_out, f, indent=2)

    # Network profiles
    net_paths = [
        base / 'live_benign' / 'mawi_network_kdd.jsonl',
        base / 'network_intrusion' / 'synthetic_500k.jsonl',
    ]

    net_rows = defaultdict(list)
    for p in net_paths:
        if not p.exists():
            continue
        def net_category(obj):
            label = obj.get('label', None)
            attack_type = str(obj.get('attack_type', '')).strip().lower()
            if label is not None and int(label) == 0:
                return 'normal'
            if attack_type:
                return attack_type
            return None
        rows, _ = sample_rows(p, args.max_samples, NET_FEATURES, net_category, max_lines=args.max_lines)
        for cat, data in rows.items():
            net_rows[cat].extend(data)

    if net_rows:
        net_profiles = build_profiles(net_rows, NET_FEATURES)
        net_out = {
            'model': 'network',
            'features': NET_FEATURES,
            'profiles': net_profiles,
        }
        with open(out_dir / 'network_profile_v1.json', 'w', encoding='utf-8') as f:
            json.dump(net_out, f, indent=2)

    print('Feature profiles written to', out_dir)


if __name__ == '__main__':
    main()
