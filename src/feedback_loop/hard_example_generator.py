"""Hard-example generation from stress-test failures."""
from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import quote, urlsplit, urlunsplit

from .failure_ingest import FailureRecord


class HardExampleGenerator:
    """Generate category-preserving hard variants for payload and URL models."""

    def __init__(self, seed: int = 42, variants_per_failure: int = 3):
        self.seed = seed
        self.variants_per_failure = variants_per_failure
        self._rng = random.Random(seed)

    def generate(self, failures: List[FailureRecord]) -> List[Dict]:
        out: List[Dict] = []
        now = datetime.now(timezone.utc).isoformat()

        for record in failures:
            for index in range(self.variants_per_failure):
                if record.model == "payload":
                    text = self._payload_variant(record, index)
                elif record.model == "url":
                    text = self._url_variant(record, index)
                else:
                    continue

                out.append(
                    {
                        "id": f"hex_{record.record_hash[:12]}_{index}",
                        "model": record.model,
                        "text": text,
                        "label": int(record.expected),
                        "category": record.category,
                        "subcategory": record.subcategory,
                        "difficulty": record.difficulty,
                        "tags": list(dict.fromkeys(record.tags + [record.category, "failure_loop"])),
                        "origin": "failure_loop",
                        "source_failure_id": record.scenario_id,
                        "run_seed": record.run_seed,
                        "generator_seed": self.seed,
                        "created_at": now,
                    }
                )

        # Deduplicate by model+text+label
        seen = set()
        unique = []
        for item in out:
            key = (item["model"], item["label"], item["text"].strip().lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    def _payload_variant(self, record: FailureRecord, index: int) -> str:
        base = record.input_preview or "test payload"
        if record.expected == 0:
            prefixes = [
                "Customer note: ",
                "Support transcript: ",
                "Documentation snippet: ",
            ]
            suffixes = ["", " // benign", " (quoted text)"]
            return f"{self._rng.choice(prefixes)}{base}{self._rng.choice(suffixes)}"

        # malicious expected=1
        techniques = [
            lambda t: quote(t),
            lambda t: t.replace(" ", "/**/"),
            lambda t: "".join(c.upper() if i % 2 == 0 else c.lower() for i, c in enumerate(t)),
            lambda t: f"{t} --{self._rng.randint(10, 99)}",
            lambda t: t.replace("OR", "O/**/R").replace("AND", "A/**/ND"),
        ]
        fn = techniques[index % len(techniques)]
        return fn(base)

    def _url_variant(self, record: FailureRecord, index: int) -> str:
        base = record.input_preview or "http://example.com"
        if not base.startswith(("http://", "https://")):
            base = f"http://{base.lstrip('/')}"

        parts = urlsplit(base)
        scheme = parts.scheme or "http"
        netloc = parts.netloc or "example.com"
        path = parts.path or "/"
        query = parts.query
        fragment = parts.fragment

        if record.expected == 0:
            safe_q = "source=docs"
            return urlunsplit((scheme, netloc, path, safe_q, fragment))

        # malicious expected=1, preserve threat-like surface
        if record.category in {"phishing", "typosquatting"}:
            netloc = netloc.replace("a", "4", 1) if "a" in netloc else f"secure-{netloc}"
            path = "/account/verify"
        elif record.category == "shorteners":
            netloc = self._rng.choice(["bit.ly", "tinyurl.com", "is.gd"]) 
            path = f"/{self._rng.choice(['Ab12xyZ', 'kL09mNo', 'pQ77rst'])}"
            query = f"url={quote(base, safe='')}"
        elif record.category == "homograph":
            netloc = netloc.replace("o", "о", 1) if "o" in netloc else netloc + "-xn"
        elif record.category == "malware":
            path = f"/download/{self._rng.choice(['update.exe', 'invoice.zip', 'payload.dll'])}"
        else:
            query = query or f"token={self._rng.randint(100000, 999999)}"

        if index % 2 == 1:
            netloc = f"{netloc}:{self._rng.choice([8080, 8443, 9001])}"

        return urlunsplit((scheme, netloc, path, query, fragment))
