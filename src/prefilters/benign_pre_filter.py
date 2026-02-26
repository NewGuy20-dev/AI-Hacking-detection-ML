"""Benign payload prefilter used to cut systematic false positives."""
import re
from typing import Callable, Optional, Tuple


class BenignPreFilter:
    """Pre-screen clearly benign payloads before running the CNN."""

    SAFE_PATTERNS = [
        re.compile(r"^GET /static/", re.IGNORECASE),
        re.compile(r"^GET /assets/", re.IGNORECASE),
        re.compile(r"^GET /favicon\.ico$", re.IGNORECASE),
        re.compile(r"^POST /api/v\d+/[a-zA-Z_/]+$", re.IGNORECASE),
        re.compile(r"^Content-Type:\s*application/json$", re.IGNORECASE),
        re.compile(r"^Authorization:\s*Bearer [A-Za-z0-9\-._~+/]+=*$", re.IGNORECASE),
    ]

    ATTACK_PATTERNS = [
        re.compile(r"<script", re.IGNORECASE),
        re.compile(r"javascript:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"\bunion\s+select\b", re.IGNORECASE),
        re.compile(r";\s*drop\s+\w+", re.IGNORECASE),
        re.compile(r"\.\./\.\.", re.IGNORECASE),
        re.compile(r"/etc/passwd", re.IGNORECASE),
        re.compile(r"\$\(", re.IGNORECASE),
    ]

    SUSPICIOUS_CHARS = set("<>;`|${}[]")

    def is_obviously_benign(self, payload: str) -> bool:
        text = str(payload).strip()
        if not text:
            return True
        return any(pattern.match(text) for pattern in self.SAFE_PATTERNS)

    def is_benign(self, payload: str) -> Tuple[bool, float, Optional[str]]:
        """Compatibility API used by existing inference wrappers."""
        text = str(payload).strip()
        if not text:
            return True, 1.0, "empty_input"

        if any(pattern.search(text) for pattern in self.ATTACK_PATTERNS):
            return False, 0.0, None

        if self.is_obviously_benign(text):
            return True, 0.98, "safe_pattern"

        if len(text) < 20 and not any(ch in self.SUSPICIOUS_CHARS for ch in text):
            return True, 0.96, "short_clean"

        if len(text) < 64 and text.replace(" ", "").isalnum():
            return True, 0.93, "short_alnum"

        return False, 0.0, None

    def get_confidence_scale(self, payload: str) -> float:
        length = len(str(payload))
        if length < 10:
            return 0.3
        if length < 20:
            return 0.5
        if length < 30:
            return 0.7
        return 1.0

    def predict(
        self,
        payload: str,
        cnn_predict: Callable[[str], Tuple[int, float]],
    ) -> Tuple[int, float]:
        """Return benign prediction for obvious safe payloads, otherwise call CNN."""
        is_benign, benign_conf, _reason = self.is_benign(payload)
        if is_benign:
            attack_prob = max(0.0, 1.0 - float(benign_conf))
            return 0, attack_prob
        return cnn_predict(payload)


_FILTER: Optional[BenignPreFilter] = None


def get_filter() -> BenignPreFilter:
    global _FILTER
    if _FILTER is None:
        _FILTER = BenignPreFilter()
    return _FILTER
