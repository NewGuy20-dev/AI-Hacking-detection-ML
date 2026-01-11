"""Benign pre-filter to reduce false positives."""
import re
from typing import Tuple, Optional


class BenignPreFilter:
    """Fast filter to bypass ML for obviously benign inputs."""
    
    # Characters commonly found in attacks
    SUSPICIOUS_CHARS = set('<>;\'"\\|`${}()[]&%')
    
    # Common safe greetings/words
    SAFE_WORDS = {
        'hi', 'hey', 'hello', 'hola', 'bye', 'goodbye',
        'thanks', 'thank', 'please', 'sorry', 'ok', 'okay',
        'yes', 'no', 'yeah', 'nope', 'sure', 'maybe',
        'good', 'great', 'nice', 'cool', 'awesome', 'love',
        'help', 'test', 'testing', 'hello world'
    }
    
    # SQL keywords that might appear in benign context
    SQL_KEYWORDS = {'select', 'drop', 'insert', 'delete', 'update', 'union', 'from', 'where'}
    
    # Patterns that look like attacks
    ATTACK_PATTERNS = [
        r'<script',
        r'javascript:',
        r'on\w+\s*=',
        r'\bor\b.*=.*',
        r'\band\b.*=.*',
        r'union\s+select',
        r';\s*drop\s',
        r';\s*delete\s',
        r'\.\./\.\.',
        r'/etc/passwd',
        r'cmd\.exe',
        r'\|\s*cat\s',
        r'\$\(',
        r'`.*`',
    ]
    
    def __init__(self):
        self._attack_regex = re.compile('|'.join(self.ATTACK_PATTERNS), re.IGNORECASE)
    
    def is_benign(self, text: str) -> Tuple[bool, float, Optional[str]]:
        """
        Check if input is obviously benign.
        
        Returns:
            (is_benign, confidence, reason)
        """
        if not text:
            return True, 1.0, "empty_input"
        
        text = str(text).strip()
        lower = text.lower()
        length = len(text)
        
        # Check for attack patterns first - if found, not benign
        if self._attack_regex.search(text):
            return False, 0.0, None
        
        # SQL keywords should NEVER be considered benign
        sql_keywords = ['select ', 'insert ', 'update ', 'delete ', 'drop ', 'union ', 'from ', 'where ', 'information_schema', 'table_name']
        if any(kw in lower for kw in sql_keywords):
            return False, 0.0, None
        
        # Very short inputs without suspicious chars are safe
        if length < 10:
            if not any(c in self.SUSPICIOUS_CHARS for c in text):
                return True, 0.98, "short_clean"
        
        # Common greetings
        words = lower.split()
        if len(words) <= 3 and words[0] in self.SAFE_WORDS:
            if not any(c in self.SUSPICIOUS_CHARS for c in text):
                return True, 0.97, "greeting"
        
        # Single safe word
        if lower.strip() in self.SAFE_WORDS:
            return True, 0.99, "safe_word"
        
        # Short alphanumeric text (like names, simple messages)
        clean_text = text.replace(' ', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '')
        if length < 30 and clean_text.isalnum():
            return True, 0.95, "short_alphanumeric"
        
        # Looks like a normal sentence (starts with capital, ends with punctuation)
        if length < 100 and text[0].isupper() and text[-1] in '.!?':
            if not any(c in self.SUSPICIOUS_CHARS for c in text):
                # Check it's not SQL-like
                if not any(kw in lower for kw in ['select ', 'drop ', 'delete ', 'insert ', 'union ']):
                    return True, 0.90, "normal_sentence"
        
        # Email-like but without suspicious patterns
        if '@' in text and '.' in text and length < 100:
            if not any(c in '<>;|`${}' for c in text):
                # Basic email pattern
                if re.match(r'^[\w\.\-]+@[\w\.\-]+\.\w+$', text):
                    return True, 0.85, "email_format"
        
        # File paths without traversal
        if ('/' in text or '\\' in text) and length < 200:
            if '..' not in text and not any(c in '<>;|`${}' for c in text):
                # Looks like a normal file path
                if re.match(r'^[A-Za-z]:\\[\w\\\.\-\s]+$', text) or re.match(r'^/[\w/\.\-]+$', text):
                    return True, 0.80, "file_path"
        
        return False, 0.0, None
    
    def get_confidence_scale(self, text: str) -> float:
        """
        Get a scaling factor for confidence based on input characteristics.
        Short/simple inputs should have reduced attack confidence.
        """
        length = len(text)
        
        if length < 5:
            return 0.2  # 80% reduction
        elif length < 10:
            return 0.3  # 70% reduction
        elif length < 20:
            return 0.5  # 50% reduction
        elif length < 30:
            return 0.7  # 30% reduction
        elif length < 50:
            return 0.85  # 15% reduction
        
        return 1.0  # No reduction for longer inputs


# Singleton instance
_filter = None

def get_filter() -> BenignPreFilter:
    global _filter
    if _filter is None:
        _filter = BenignPreFilter()
    return _filter
