"""Edge case generators for stress testing."""
import random
import string
from typing import List, Iterator
from datetime import date
import hashlib


class LongInputGenerator:
    """Generate very long inputs (10k+ characters)."""
    
    WORDS = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", "it", "for",
             "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his"]
    
    def generate_repeated_char(self, char: str = "A", length: int = 10000) -> str:
        return char * length
    
    def generate_repeated_word(self, word: str = "test", length: int = 10000) -> str:
        repeats = length // (len(word) + 1)
        return " ".join([word] * repeats)
    
    def generate_lorem(self, length: int = 10000) -> str:
        words = []
        current_len = 0
        while current_len < length:
            word = random.choice(self.WORDS)
            words.append(word)
            current_len += len(word) + 1
        return " ".join(words)[:length]
    
    def generate_nested_html(self, depth: int = 500) -> str:
        return "<div>" * depth + "content" + "</div>" * depth
    
    def generate_long_sql(self, columns: int = 1000) -> str:
        cols = ", ".join([f"col{i}" for i in range(columns)])
        return f"SELECT {cols} FROM table"
    
    def generate_all(self, count: int = 10) -> List[str]:
        samples = [
            self.generate_repeated_char("A", 10000),
            self.generate_repeated_char("X", 15000),
            self.generate_repeated_word("test", 10000),
            self.generate_lorem(10000),
            self.generate_lorem(20000),
            self.generate_nested_html(500),
            self.generate_nested_html(1000),
            self.generate_long_sql(500),
            self.generate_long_sql(1000),
            " ".join(["word"] * 5000),
        ]
        return samples[:count]


class UnicodeGenerator:
    """Generate unicode and emoji edge cases."""
    
    # Various unicode categories
    EMOJIS = ["👋", "🌍", "❤️", "🔥", "✨", "🎉", "💯", "🚀", "💻", "🔒", "⚠️", "✅", "❌"]
    COMPOUND_EMOJIS = ["👨‍👩‍👧‍👦", "🏳️‍🌈", "👩‍💻", "🧑‍🔬", "👨‍🚀"]
    RTL_CHARS = ["א", "ב", "ג", "ד", "ה", "و", "ي", "م", "ن"]
    COMBINING_CHARS = ["\u0300", "\u0301", "\u0302", "\u0303", "\u0304", "\u0305"]
    SPECIAL_UNICODE = [
        "\u202e",  # RTL override
        "\u200b",  # Zero-width space
        "\u200c",  # Zero-width non-joiner
        "\u200d",  # Zero-width joiner
        "\ufeff",  # BOM
        "\u00a0",  # Non-breaking space
    ]
    SCRIPTS = {
        'arabic': "مرحبا بالعالم",
        'chinese': "你好世界",
        'russian': "Привет мир",
        'japanese': "こんにちは世界",
        'korean': "안녕하세요 세계",
        'greek': "Γειά σου κόσμε",
        'hebrew': "שלום עולם",
        'thai': "สวัสดีโลก",
    }
    
    def generate_emoji_text(self) -> str:
        emojis = random.sample(self.EMOJIS, min(5, len(self.EMOJIS)))
        return f"Hello {''.join(emojis)} World"
    
    def generate_compound_emoji(self) -> str:
        return f"Family: {random.choice(self.COMPOUND_EMOJIS)} together"
    
    def generate_multi_script(self) -> str:
        scripts = random.sample(list(self.SCRIPTS.values()), 3)
        return " | ".join(scripts)
    
    def generate_combining_chars(self) -> str:
        base = "test"
        result = []
        for c in base:
            result.append(c)
            for _ in range(random.randint(1, 3)):
                result.append(random.choice(self.COMBINING_CHARS))
        return ''.join(result)
    
    def generate_rtl_mixed(self) -> str:
        rtl = ''.join(random.sample(self.RTL_CHARS, 5))
        return f"LTR text {rtl} more LTR"
    
    def generate_rtl_override(self) -> str:
        return f"\u202edesrever si txet siht"
    
    def generate_zero_width(self) -> str:
        text = "normal"
        result = []
        for c in text:
            result.append(c)
            result.append(random.choice(["\u200b", "\u200c", "\u200d"]))
        return ''.join(result)
    
    def generate_bom(self) -> str:
        return "\ufeff" + "text with BOM"
    
    def generate_all(self, count: int = 20) -> List[str]:
        samples = [
            self.generate_emoji_text(),
            self.generate_compound_emoji(),
            self.generate_multi_script(),
            self.generate_combining_chars(),
            self.generate_rtl_mixed(),
            self.generate_rtl_override(),
            self.generate_zero_width(),
            self.generate_bom(),
            "".join(self.EMOJIS),
            "🔥" * 100,
            self.SCRIPTS['chinese'] + self.SCRIPTS['arabic'],
            f"Mix: {self.SCRIPTS['russian']} and {self.SCRIPTS['japanese']}",
        ]
        # Add more random combinations
        while len(samples) < count:
            samples.append(self.generate_emoji_text())
            samples.append(self.generate_multi_script())
        return samples[:count]


class PolyglotGenerator:
    """Generate polyglot payloads valid in multiple contexts."""
    
    POLYGLOTS = [
        # JS + HTML
        "javascript:/*--></title></style></textarea></script><svg onload='+/\"/+/onmouseover=1/+/[*/[]/+alert(1)//'>",
        # SQL + XSS
        "1'><script>alert(1)</script>--",
        # Command + SQL
        "'; cat /etc/passwd; --",
        # Multiple contexts
        "{{constructor.constructor('return this')()}}", 
        # SSTI + XSS
        "{{7*7}}<script>alert(1)</script>",
        # JSON + JS
        '{"__proto__":{"isAdmin":true}}',
        # XML + XSS
        "<![CDATA[<script>alert(1)</script>]]>",
        # CSS + JS
        "expression(alert(1))",
        # URL + command
        "file:///etc/passwd",
        # Multiple encodings
        "%253Cscript%253Ealert(1)%253C/script%253E",
    ]
    
    def generate_all(self) -> List[str]:
        return self.POLYGLOTS.copy()


class SpecialCharGenerator:
    """Generate special character edge cases."""
    
    def generate_null_bytes(self) -> List[str]:
        return [
            "test\x00hidden",
            "\x00start",
            "end\x00",
            "a\x00b\x00c",
            "SELECT\x00*\x00FROM",
        ]
    
    def generate_newlines(self) -> List[str]:
        return [
            "line1\nline2",
            "line1\r\nline2",
            "line1\rline2",
            "multi\n\n\nlines",
            "tab\there",
        ]
    
    def generate_control_chars(self) -> List[str]:
        return [
            "bell\x07char",
            "backspace\x08test",
            "escape\x1bseq",
            "form\x0cfeed",
            "vertical\x0btab",
        ]
    
    def generate_quotes(self) -> List[str]:
        return [
            'single\'quote',
            'double"quote',
            'back`tick',
            "mixed'\"quotes",
            "escaped\\'quote",
            'nested "\'quotes\'"',
        ]
    
    def generate_all(self) -> List[str]:
        samples = []
        samples.extend(self.generate_null_bytes())
        samples.extend(self.generate_newlines())
        samples.extend(self.generate_control_chars())
        samples.extend(self.generate_quotes())
        return samples


class EdgeCaseGenerator:
    """Main generator combining all edge case types."""
    
    def __init__(self, seed: int = None):
        if seed is None:
            seed = int(hashlib.md5(date.today().isoformat().encode()).hexdigest()[:8], 16)
        random.seed(seed)
        self.seed = seed
        
        self.long_input = LongInputGenerator()
        self.unicode = UnicodeGenerator()
        self.polyglot = PolyglotGenerator()
        self.special = SpecialCharGenerator()
    
    def generate_all(self, count_per_category: int = 10) -> dict:
        """Generate all edge cases by category."""
        return {
            'long_inputs': self.long_input.generate_all(count_per_category),
            'unicode': self.unicode.generate_all(count_per_category),
            'polyglots': self.polyglot.generate_all(),
            'special_chars': self.special.generate_all(),
        }
    
    def generate_flat(self, total_count: int = 100) -> List[str]:
        """Generate flat list of edge cases."""
        all_cases = self.generate_all()
        flat = []
        for cases in all_cases.values():
            flat.extend(cases)
        random.shuffle(flat)
        return flat[:total_count]


# Convenience function
def get_daily_edge_cases() -> EdgeCaseGenerator:
    """Get generator with today's seed."""
    return EdgeCaseGenerator()


if __name__ == "__main__":
    gen = EdgeCaseGenerator(seed=42)
    
    cases = gen.generate_all(count_per_category=3)
    for category, samples in cases.items():
        print(f"\n{category.upper()}:")
        for s in samples[:3]:
            preview = s[:50] + "..." if len(s) > 50 else s
            print(f"  [{len(s)} chars] {repr(preview)}")
