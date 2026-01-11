"""Context-aware classifier for reducing false positives based on input context."""
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class InputContext(Enum):
    """Types of input contexts."""
    EMAIL_FIELD = "email_field"
    SEARCH_QUERY = "search_query"
    COMMENT = "comment"
    CODE_FILE = "code_file"
    CONFIG_FILE = "config_file"
    SQL_EDITOR = "sql_editor"
    CHAT_MESSAGE = "chat_message"
    FORM_INPUT = "form_input"
    URL_PARAM = "url_param"
    FILE_PATH = "file_path"
    UNKNOWN = "unknown"


@dataclass
class ContextResult:
    """Result of context-aware classification."""
    original_score: float
    adjusted_score: float
    context: InputContext
    adjustment_reason: Optional[str]
    is_fp_candidate: bool


class ContextAwareClassifier:
    """Adjust classification scores based on input context to reduce false positives."""
    
    # Patterns that indicate specific contexts
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    FILE_PATH_PATTERN = re.compile(r'^([A-Za-z]:\\|/|\.\.?/|~/)')
    URL_PATTERN = re.compile(r'^https?://')
    
    # Benign patterns that often trigger FPs
    HEART_EMOJI = re.compile(r'<3(?!\w)')  # <3 not followed by alphanumeric
    MENU_SQL = re.compile(r'\b(SELECT|FROM)\b.*\b(menu|dish|food|appetizer|entree|dessert|wine|breakfast|lunch|dinner)\b', re.I)
    PRICING_OR = re.compile(r'\$\d+.*\bOR\b.*\b(offer|trade|best|nearest)\b', re.I)
    EMOTICON_LT = re.compile(r'<[3D\)\(PpOo]')  # Common emoticons starting with <
    
    # Context-specific whitelist patterns
    CONTEXT_WHITELISTS = {
        InputContext.EMAIL_FIELD: [
            re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        ],
        InputContext.SEARCH_QUERY: [
            re.compile(r'^[\w\s\-.,!?]+$'),  # Simple text queries
        ],
        InputContext.COMMENT: [
            re.compile(r'<3'),  # Heart emoji in comments
            re.compile(r'^\w[\w\s.,!?\'"()-]*$'),  # Normal comment text
        ],
        InputContext.CODE_FILE: [
            re.compile(r'(SELECT|INSERT|UPDATE|DELETE)\s+', re.I),  # SQL in code
            re.compile(r'<script'),  # Script tags in code
        ],
        InputContext.SQL_EDITOR: [
            re.compile(r'.*'),  # Allow all in SQL editor context
        ],
    }
    
    # Score adjustments per context (negative = reduce score)
    CONTEXT_ADJUSTMENTS = {
        InputContext.EMAIL_FIELD: -0.4,
        InputContext.SEARCH_QUERY: -0.2,
        InputContext.COMMENT: -0.3,
        InputContext.CODE_FILE: -0.5,
        InputContext.SQL_EDITOR: -0.8,
        InputContext.CHAT_MESSAGE: -0.2,
        InputContext.FILE_PATH: -0.3,
    }
    
    def detect_context(self, text: str, hint: Optional[str] = None) -> InputContext:
        """Auto-detect input context from text patterns."""
        if hint:
            try:
                return InputContext(hint.lower())
            except ValueError:
                pass
        
        text_stripped = text.strip()
        
        # Check for email
        if self.EMAIL_PATTERN.match(text_stripped):
            return InputContext.EMAIL_FIELD
        
        # Check for file path
        if self.FILE_PATH_PATTERN.match(text_stripped):
            return InputContext.FILE_PATH
        
        # Check for URL
        if self.URL_PATTERN.match(text_stripped):
            return InputContext.URL_PARAM
        
        # Check for code patterns
        if any(p in text for p in ['def ', 'function ', 'class ', 'import ', 'const ', 'var ', 'let ']):
            return InputContext.CODE_FILE
        
        # Check for chat/comment patterns
        if self.HEART_EMOJI.search(text) or self.EMOTICON_LT.search(text):
            return InputContext.CHAT_MESSAGE
        
        return InputContext.UNKNOWN
    
    def is_benign_pattern(self, text: str) -> Tuple[bool, Optional[str]]:
        """Check if text matches known benign patterns that trigger FPs."""
        # Heart emoji
        if self.HEART_EMOJI.search(text):
            return True, "Heart emoji (<3) detected"
        
        # Menu/restaurant SQL-like text
        if self.MENU_SQL.search(text):
            return True, "Menu/restaurant context with SQL keywords"
        
        # Pricing with OR
        if self.PRICING_OR.search(text):
            return True, "Pricing text with OR keyword"
        
        # Email address
        if self.EMAIL_PATTERN.match(text.strip()):
            return True, "Valid email address format"
        
        # Emoticons
        if self.EMOTICON_LT.search(text) and len(text) < 50:
            return True, "Emoticon pattern detected"
        
        return False, None
    
    def adjust_score(self, text: str, score: float, 
                    context_hint: Optional[str] = None) -> ContextResult:
        """Adjust classification score based on context."""
        context = self.detect_context(text, context_hint)
        original_score = score
        adjustment_reason = None
        is_fp_candidate = False
        
        # Check for known benign patterns
        is_benign, reason = self.is_benign_pattern(text)
        if is_benign:
            # Significant reduction for known benign patterns
            score = max(0.0, score - 0.5)
            adjustment_reason = reason
            is_fp_candidate = True
        
        # Apply context-based adjustment
        if context in self.CONTEXT_ADJUSTMENTS:
            adj = self.CONTEXT_ADJUSTMENTS[context]
            score = max(0.0, min(1.0, score + adj))
            if not adjustment_reason:
                adjustment_reason = f"Context adjustment for {context.value}"
        
        # Check context-specific whitelists
        if context in self.CONTEXT_WHITELISTS:
            for pattern in self.CONTEXT_WHITELISTS[context]:
                if pattern.search(text):
                    score = max(0.0, score - 0.3)
                    is_fp_candidate = True
                    if not adjustment_reason:
                        adjustment_reason = f"Whitelist match for {context.value}"
                    break
        
        return ContextResult(
            original_score=original_score,
            adjusted_score=round(score, 4),
            context=context,
            adjustment_reason=adjustment_reason,
            is_fp_candidate=is_fp_candidate
        )
    
    def classify_with_context(self, text: str, base_score: float,
                             context_hint: Optional[str] = None,
                             threshold: float = 0.5) -> Dict:
        """Full classification with context awareness."""
        result = self.adjust_score(text, base_score, context_hint)
        
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "original_score": result.original_score,
            "adjusted_score": result.adjusted_score,
            "context": result.context.value,
            "is_malicious": result.adjusted_score >= threshold,
            "is_fp_candidate": result.is_fp_candidate,
            "adjustment_reason": result.adjustment_reason,
            "score_change": round(result.adjusted_score - result.original_score, 4),
        }


def apply_context_filter(text: str, score: float, 
                        context: Optional[str] = None) -> float:
    """Convenience function to apply context-aware filtering."""
    classifier = ContextAwareClassifier()
    result = classifier.adjust_score(text, score, context)
    return result.adjusted_score


# Pre-instantiated classifier for quick access
_classifier = ContextAwareClassifier()


def quick_adjust(text: str, score: float) -> float:
    """Quick score adjustment without full result object."""
    return _classifier.adjust_score(text, score).adjusted_score
