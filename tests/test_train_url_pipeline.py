import json
from pathlib import Path

from src.training.train_url import (
    generate_benign_hard_negative_urls,
    generate_malicious_shortener_urls,
    load_url_data,
)


def _write_lines(path: Path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_url_data_reports_exact_source_counts_and_normalizes_urls(tmp_path):
    base = tmp_path
    url_dir = base / "datasets" / "url_analysis"
    live_benign_dir = base / "datasets" / "live_benign"

    _write_lines(
        url_dir / "real_malicious_urls.txt",
        [
            "http://bad-one.test/login",
            "https://bad-two.test/payload",
            "not-a-url",
        ],
    )
    _write_lines(
        url_dir / "kaggle_malicious_urls.csv",
        [
            "url,label",
            "evil-kaggle.test,1",
            "https://good-kaggle.test,0",
            "invalid-row",
        ],
    )
    _write_lines(
        url_dir / "synthetic_malicious_hard.txt",
        [
            "synthetic-bad-one.test/dropper",
            "",
            "http://synthetic-bad-two.test/phish",
        ],
    )

    cc_path = live_benign_dir / "common_crawl_urls.jsonl"
    cc_path.parent.mkdir(parents=True, exist_ok=True)
    with cc_path.open("w", encoding="utf-8") as handle:
        for idx in range(995):
            handle.write(json.dumps({"text": f"https://benign-{idx}.example/path"}) + "\n")
        handle.write('{"text": ""}\n')
        handle.write("not-json\n")

    _write_lines(
        url_dir / "top-1m.csv",
        [
            "1,tranco-one.example",
            "2,tranco-two.example",
        ],
    )
    _write_lines(
        url_dir / "synthetic_benign_hard.txt",
        [
            "https://benign-hard-one.example",
            "",
            "benign-hard-two.example/about",
        ],
    )
    _write_lines(
        base / "datasets" / "curated_benign" / "adversarial" / "url_benign.txt",
        [
            "http://127.0.0.1:8080",
            "tinyurl.com/benign-help",
        ],
    )

    hard_examples = base / "hard_examples.jsonl"
    hard_examples.write_text(
        json.dumps({"model": "url", "text": "hard-malicious.example", "label": 1}) + "\n"
        + json.dumps({"model": "url", "text": "https://hard-benign.example", "label": 0}) + "\n"
        + json.dumps({"model": "payload", "text": "ignored.example", "label": 1}) + "\n",
        encoding="utf-8",
    )

    urls, labels, summary = load_url_data(
        base,
        hard_examples_file=str(hard_examples),
        return_summary=True,
        generated_hard_negative_count=0,
        generated_shortener_attack_count=0,
        curated_adversarial_limit=2,
    )

    assert summary["source_counts"]["urlhaus_real_malicious"] == {
        "total": 2,
        "malicious": 2,
        "benign": 0,
    }
    assert summary["source_counts"]["kaggle_csv"] == {
        "total": 2,
        "malicious": 1,
        "benign": 1,
    }
    assert summary["source_counts"]["synthetic_malicious_hard"] == {
        "total": 2,
        "malicious": 2,
        "benign": 0,
    }
    assert summary["source_counts"]["common_crawl_urls"] == {
        "total": 995,
        "malicious": 0,
        "benign": 995,
    }
    assert summary["source_counts"]["tranco_top_domains"] == {
        "total": 2,
        "malicious": 0,
        "benign": 2,
    }
    assert summary["source_counts"]["synthetic_benign_hard"] == {
        "total": 2,
        "malicious": 0,
        "benign": 2,
    }
    assert summary["source_counts"]["curated_adversarial_url_benign"] == {
        "total": 2,
        "malicious": 0,
        "benign": 2,
    }
    assert summary["source_counts"]["hard_examples"] == {
        "total": 6,
        "malicious": 3,
        "benign": 3,
    }
    assert summary["skipped_counts"] == {
        "synthetic_malicious_hard": 1,
        "common_crawl_urls": 2,
        "synthetic_benign_hard": 1,
    }
    assert summary["hard_examples_repeat_factor"] == 3

    assert len(urls) == 1013
    assert len(labels) == 1013
    assert summary["totals"] == {
        "total": 1013,
        "malicious": 8,
        "benign": 1005,
    }

    assert "http://evil-kaggle.test" in urls
    assert "http://synthetic-bad-one.test/dropper" in urls
    assert "https://tranco-one.example/" in urls
    assert "http://hard-malicious.example" in urls
    assert "http://tinyurl.com/benign-help" in urls
    assert all(url.strip() for url in urls)
    assert set(labels) == {0, 1}


def test_generate_benign_hard_negative_urls_contains_ip_literals_and_shorteners():
    urls = generate_benign_hard_negative_urls(12)

    assert len(urls) == 12
    assert any("bit.ly/" in url or "tinyurl.com/" in url or "is.gd/" in url or "t.co/" in url or "cutt.ly/" in url for url in urls)
    assert any(
        "://" in url and len(url.split("://", 1)[1].split("/", 1)[0].split(".")) == 4
        for url in urls
    )
    assert any("redirect=" in url for url in urls)


def test_generate_malicious_shortener_urls_contains_redirect_attacks():
    urls = generate_malicious_shortener_urls(12)

    assert len(urls) == 12
    assert any("bit.ly/" in url or "tinyurl.com/" in url or "is.gd/" in url or "t.co/" in url or "cutt.ly/" in url for url in urls)
    assert any("?url=" in url or "?redirect=" in url for url in urls)
    assert any("malicious-redirect-" in url for url in urls)
