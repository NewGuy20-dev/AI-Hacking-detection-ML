"""Adversarial sample generators for stress testing."""
import random
import base64
import urllib.parse
from typing import List, Iterator
from datetime import date
import hashlib


# Character substitution mappings
CHAR_SUBS = {
    'a': ['@', '4', 'α', 'а'],  # Last is Cyrillic
    'e': ['3', 'є', 'е', 'ё'],
    'i': ['1', '!', 'і', 'ι'],
    'o': ['0', 'ο', 'о', '°'],
    's': ['$', '5', 'ѕ', '§'],
    'l': ['1', '|', 'ӏ', 'ℓ'],
    't': ['+', '†', 'т'],
    'c': ['(', '¢', 'с'],
    'n': ['и', 'ñ'],
    'b': ['8', 'ь', 'β'],
    'g': ['9', 'ğ'],
    'u': ['υ', 'ü', 'µ'],
}

# SQL keywords for obfuscation
SQL_KEYWORDS = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 'WHERE', 'FROM', 'OR', 'AND']


class CharSubstitutionEngine:
    """Generate character substitution variants."""
    
    def __init__(self, intensity: float = 0.3):
        self.intensity = intensity
    
    def substitute(self, text: str) -> str:
        result = []
        for c in text:
            if c.lower() in CHAR_SUBS and random.random() < self.intensity:
                result.append(random.choice(CHAR_SUBS[c.lower()]))
            else:
                result.append(c)
        return ''.join(result)
    
    def generate_variants(self, text: str, count: int = 5) -> List[str]:
        return [self.substitute(text) for _ in range(count)]


class EncodingVariationEngine:
    """Generate encoding variations of payloads."""
    
    @staticmethod
    def url_encode(text: str) -> str:
        return urllib.parse.quote(text, safe='')
    
    @staticmethod
    def double_url_encode(text: str) -> str:
        return urllib.parse.quote(urllib.parse.quote(text, safe=''), safe='')
    
    @staticmethod
    def base64_encode(text: str) -> str:
        return base64.b64encode(text.encode()).decode()
    
    @staticmethod
    def hex_encode(text: str) -> str:
        return ''.join(f'\\x{ord(c):02x}' for c in text)
    
    @staticmethod
    def unicode_encode(text: str) -> str:
        return ''.join(f'\\u{ord(c):04x}' for c in text)
    
    @staticmethod
    def html_entity_encode(text: str) -> str:
        return ''.join(f'&#{ord(c)};' for c in text)
    
    def generate_all(self, text: str) -> List[str]:
        return [
            text,
            self.url_encode(text),
            self.double_url_encode(text),
            self.base64_encode(text),
            self.hex_encode(text),
            self.unicode_encode(text),
            self.html_entity_encode(text),
        ]


class CaseVariationEngine:
    """Generate case variations."""
    
    @staticmethod
    def alternating(text: str) -> str:
        return ''.join(c.upper() if i % 2 else c.lower() for i, c in enumerate(text))
    
    @staticmethod
    def random_case(text: str) -> str:
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in text)
    
    @staticmethod
    def inverse_alternating(text: str) -> str:
        return ''.join(c.lower() if i % 2 else c.upper() for i, c in enumerate(text))
    
    def generate_all(self, text: str) -> List[str]:
        return [
            text.upper(),
            text.lower(),
            self.alternating(text),
            self.random_case(text),
            self.inverse_alternating(text),
            text.capitalize(),
        ]


class ObfuscationEngine:
    """Generate obfuscated SQL/code variants."""
    
    @staticmethod
    def add_comments(text: str) -> str:
        """Add SQL comments between characters."""
        result = []
        for i, c in enumerate(text):
            result.append(c)
            if c.isalpha() and i < len(text) - 1 and text[i + 1].isalpha():
                if random.random() < 0.3:
                    result.append('/**/')
        return ''.join(result)
    
    @staticmethod
    def add_whitespace(text: str) -> str:
        """Add extra whitespace."""
        spaces = [' ', '\t', '\n', '  ', '   ']
        result = []
        for c in text:
            result.append(c)
            if c == ' ' and random.random() < 0.5:
                result.append(random.choice(spaces))
        return ''.join(result)
    
    @staticmethod
    def use_concat(text: str) -> str:
        """Use string concatenation."""
        if len(text) < 4:
            return text
        mid = len(text) // 2
        return f"'{text[:mid]}'+'{text[mid:]}'"
    
    @staticmethod
    def use_char_func(text: str) -> str:
        """Convert to CHAR() function calls."""
        return '+'.join(f'CHAR({ord(c)})' for c in text[:20])  # Limit length
    
    def generate_all(self, text: str) -> List[str]:
        return [
            text,
            self.add_comments(text),
            self.add_whitespace(text),
            self.use_concat(text),
            self.use_char_func(text) if len(text) <= 20 else text,
        ]


class AdversarialGenerator:
    """Main generator combining all adversarial techniques."""
    
    def __init__(self, seed: int = None):
        if seed is None:
            # Use date-based seed for daily uniqueness
            seed = int(hashlib.md5(date.today().isoformat().encode()).hexdigest()[:8], 16)
        random.seed(seed)
        self.seed = seed
        
        self.char_sub = CharSubstitutionEngine()
        self.encoding = EncodingVariationEngine()
        self.case_var = CaseVariationEngine()
        self.obfuscation = ObfuscationEngine()
    
    def generate_from_payload(self, payload: str) -> List[str]:
        """Generate all adversarial variants of a payload."""
        variants = set()
        variants.add(payload)
        
        # Character substitutions
        variants.update(self.char_sub.generate_variants(payload, 3))
        
        # Encoding variations
        variants.update(self.encoding.generate_all(payload))
        
        # Case variations
        variants.update(self.case_var.generate_all(payload))
        
        # Obfuscation (for SQL-like payloads)
        if any(kw in payload.upper() for kw in SQL_KEYWORDS):
            variants.update(self.obfuscation.generate_all(payload))
        
        return list(variants)
    
    def generate_batch(self, payloads: List[str], variants_per_payload: int = 10) -> Iterator[str]:
        """Generate adversarial variants for a batch of payloads."""
        for payload in payloads:
            all_variants = self.generate_from_payload(payload)
            random.shuffle(all_variants)
            for v in all_variants[:variants_per_payload]:
                yield v
    
    def generate_benign_adversarial(self, count: int = 1000) -> Iterator[str]:
        """Generate benign samples that might look suspicious."""
        templates = [
            # SQL-like but benign
            "SELECT your favorite color",
            "Please select from the menu",
            "Drop by anytime!",
            "Drop me a line",
            "Union of workers",
            "Delete old files manually",
            "Update your profile",
            "Insert coin to continue",
            "WHERE are you going?",
            "FROM: John Smith",
            "OR you could try this",
            "AND another thing...",
            
            # Script-like but benign
            "<3 love this",
            "I <3 you",
            "5 > 3 is true",
            "a < b comparison",
            "Use /bin/bash for scripts",
            "C:\\Users\\Admin\\Documents",
            "alert('Hello!')",  # In quotes, discussing code
            "The onclick event fires",
            
            # Code discussion
            "def function(): pass",
            "class MyClass:",
            "import os",
            "SELECT * FROM users -- this is a comment",
            "eval() is dangerous",
            "exec() should be avoided",
        ]
        
        generated = 0
        while generated < count:
            template = random.choice(templates)
            
            # Apply random transformations
            if random.random() < 0.3:
                template = self.char_sub.substitute(template)
            if random.random() < 0.2:
                template = random.choice(self.case_var.generate_all(template))
            
            yield template
            generated += 1


# Convenience function
def get_daily_generator() -> AdversarialGenerator:
    """Get generator with today's seed."""
    return AdversarialGenerator()


if __name__ == "__main__":
    gen = AdversarialGenerator(seed=42)
    
    # Test payload variants
    payload = "' OR '1'='1"
    print(f"Original: {payload}")
    print("Variants:")
    for v in gen.generate_from_payload(payload)[:10]:
        print(f"  {v}")
    
    # Test benign adversarial
    print("\nBenign adversarial samples:")
    for sample in list(gen.generate_benign_adversarial(5)):
        print(f"  {sample}")
