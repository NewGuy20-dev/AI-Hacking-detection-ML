#!/usr/bin/env python3
"""Analyze URL dataset/failure composition by pattern category."""
import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


REQUIRED_BENIGN_CATEGORIES = (
    "cdn",
    "oauth",
    "api",
    "saas",
    "shortener",
    "login",
    "query_params",
)


def _categorize(url: str) -> set[str]:
    text = str(url).strip()
    categories: set[str] = set()
    if not text:
        return categories
    lower = text.lower()
    parts = urlsplit(text if text.startswith(("http://", "https://")) else f"http://{text}")
    host = parts.netloc.lower()
    path = parts.path.lower()
    query = parts.query.lower()

    if "cdn" in host or "cloudfront" in host or "jsdelivr" in host:
        categories.add("cdn")
    if "oauth" in lower or "microsoftonline.com" in host or "accounts.google.com" in host:
        categories.add("oauth")
    if host.startswith("api.") or "/api/" in path:
        categories.add("api")
    if any(k in host for k in ("github.com", "atlassian.net", "slack.com", "notion.so", "salesforce.com")):
        categories.add("saas")
    if any(k in host for k in ("bit.ly", "t.co", "tinyurl.com", "cutt.ly", "is.gd")):
        categories.add("shortener")
    if any(k in path for k in ("/login", "/signin", "/auth", "/account")):
        categories.add("login")
    if query:
        categories.add("query_params")
    if not categories:
        categories.add("other")
    return categories


def _iter_urls_from_file(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                text = obj.get("url") or obj.get("text") or obj.get("input_preview") or ""
                if text:
                    yield str(text).strip()
    elif suffix == ".csv":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f):
                if idx == 0 and "," in line.lower():
                    continue
                value = line.strip().split(",")[0].strip().strip('"')
                if value:
                    yield value
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def _collect_urls(split: str) -> list[str]:
    root = Path("datasets/url_analysis")
    if split == "benign":
        candidates = [
            root / "synthetic_benign_hard.txt",
            root / "url_benign_expansion.jsonl",
            root / "top-1m.csv",
            root / "domains" / "top-1m.csv",
        ]
    else:
        candidates = [
            root / "synthetic_malicious_hard.txt",
            root / "url_malicious_expansion.jsonl",
            root / "real_malicious_urls.txt",
            root / "urlhaus.csv",
        ]

    urls: list[str] = []
    for file_path in candidates:
        if file_path.exists():
            urls.extend(list(_iter_urls_from_file(file_path)))
    return urls


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze URL dataset composition.")
    parser.add_argument("--split", choices=["benign", "malicious"], default="benign")
    parser.add_argument("--show-categories", action="store_true")
    parser.add_argument("--failures-file", type=str, default=None)
    parser.add_argument("--sample-limit", type=int, default=200000)
    args = parser.parse_args()

    urls: list[str] = []
    if args.failures_file:
        fail_path = Path(args.failures_file)
        if not fail_path.exists():
            raise FileNotFoundError(f"Missing failures file: {fail_path}")
        with open(fail_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if int(obj.get("expected", 0)) == 0 and int(obj.get("predicted", 1)) == 1:
                    urls.append(str(obj.get("input_preview", "")).strip())
    else:
        urls = _collect_urls(args.split)

    if args.sample_limit > 0:
        urls = urls[: args.sample_limit]

    category_counts = Counter()
    for url in urls:
        for category in _categorize(url):
            category_counts[category] += 1

    print(f"Analyzed {len(urls)} URLs")
    for name, count in category_counts.most_common():
        print(f"{name:14s} {count}")

    if args.show_categories and args.split == "benign":
        missing = [cat for cat in REQUIRED_BENIGN_CATEGORIES if category_counts[cat] == 0]
        if missing:
            print("\nMissing benign categories:", ", ".join(missing))
        else:
            print("\nAll required benign categories are present.")


if __name__ == "__main__":
    main()
