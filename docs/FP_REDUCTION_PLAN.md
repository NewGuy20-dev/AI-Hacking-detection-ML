# False Positive Reduction Plan

## Problem Analysis

The model has a **critical false positive problem** where benign inputs like "Hey" are flagged as threats with 95%+ confidence.

### Root Causes Identified:

1. **Training Data Imbalance**: Model trained predominantly on malicious examples
2. **Low Confidence Threshold**: Current threshold is 0.40 for payload_cnn (way too low)
3. **Character-level Overfitting**: CNN learns spurious patterns from common characters
4. **No Benign Pre-filter**: Every input goes through the full attack detection pipeline
5. **Uncalibrated Confidence**: Raw sigmoid outputs don't reflect true probability

### Evidence from Validation Report:
- "john.doe@example.com" → 100% confident threat ❌
- "O'Brien" → 99.95% confident threat ❌  
- "The script was great!" → 99.95% confident threat ❌
- "C:\Users\Admin" → 100% confident threat ❌
- "<3 love this product" → 85.6% confident threat ❌

---

## Solution Plan (Priority Order)

### Phase 1: Immediate Fixes (1-2 hours)

#### 1.1 Raise Confidence Threshold
Change from 0.40 to **0.85** for payload classification.

```python
# In configs/optimal_thresholds.json
"payload_cnn": 0.85,  # Was 0.40
"url_cnn": 0.80,      # Was 0.45
```

#### 1.2 Add Benign Pre-filter
Create a fast pre-filter that bypasses ML for obviously benign inputs:

```python
class BenignPreFilter:
    """Fast filter to skip ML for obviously benign inputs."""
    
    # Patterns that are NEVER attacks
    SAFE_PATTERNS = [
        # Short greetings (< 20 chars, no special chars)
        lambda x: len(x) < 20 and x.isalnum() or x.replace(' ', '').isalnum(),
        # Common greetings
        lambda x: x.lower().strip() in ['hi', 'hey', 'hello', 'thanks', 'ok', 'yes', 'no'],
        # Simple sentences without suspicious chars
        lambda x: len(x) < 50 and not any(c in x for c in ['<', '>', ';', '|', '`', '$', '{', '}']),
    ]
    
    # Characters that indicate potential attack
    SUSPICIOUS_CHARS = set('<>;\'"\\|`${}()[]')
    
    def is_obviously_benign(self, text: str) -> bool:
        text = str(text).strip()
        
        # Very short alphanumeric = safe
        if len(text) < 15 and text.replace(' ', '').replace('.', '').isalnum():
            return True
        
        # No suspicious characters and short = safe
        if len(text) < 30 and not any(c in self.SUSPICIOUS_CHARS for c in text):
            return True
        
        # Common greeting patterns
        lower = text.lower()
        if lower in ['hi', 'hey', 'hello', 'thanks', 'thank you', 'ok', 'okay', 'yes', 'no', 'bye']:
            return True
            
        return False
```

#### 1.3 Add Length-based Confidence Scaling
Very short inputs should have reduced confidence:

```python
def scale_confidence_by_length(confidence: float, text: str) -> float:
    """Reduce confidence for very short inputs."""
    length = len(text)
    if length < 10:
        return confidence * 0.3  # 70% reduction
    elif length < 20:
        return confidence * 0.5  # 50% reduction
    elif length < 30:
        return confidence * 0.7  # 30% reduction
    return confidence
```

---

### Phase 2: Model Improvements (4-8 hours)

#### 2.1 Retrain with Balanced Dataset
Current issue: Too many malicious examples, not enough benign.

**Target ratio**: 60% benign, 40% malicious

**Benign data sources to add**:
- Common English sentences (10M+)
- Email subjects and bodies
- Chat messages / social media posts
- Code comments (non-malicious)
- File paths (Windows/Linux)
- Names with apostrophes (O'Brien, etc.)
- URLs with common query params

#### 2.2 Add Negative Mining
Find and add examples the model gets wrong:

```python
# Generate hard negatives
hard_negatives = [
    "The script was great!",  # Contains "script"
    "SELECT your favorite menu item",  # Contains "SELECT"
    "DROP by anytime",  # Contains "DROP"
    "O'Brien's restaurant",  # Contains quote
    "C:\\Users\\Documents",  # Contains backslash
    "john.doe@example.com",  # Email format
    "<3 love this",  # Contains <
    "Price: $100",  # Contains $
]
```

#### 2.3 Implement Confidence Calibration
Use temperature scaling to calibrate confidence:

```python
class CalibratedPredictor:
    def __init__(self, model, temperature=2.0):
        self.model = model
        self.temperature = temperature  # Higher = less confident
    
    def predict(self, x):
        logits = self.model(x)
        # Apply temperature scaling
        calibrated_logits = logits / self.temperature
        return torch.sigmoid(calibrated_logits)
```

---

### Phase 3: Architecture Changes (1-2 days)

#### 3.1 Two-Stage Classification

```
Input → [Benign Pre-filter] → Obviously Safe? → Return SAFE
                ↓ No
        [Quick Heuristics] → No suspicious patterns? → Return SAFE  
                ↓ Has patterns
        [ML Model] → Confidence > 0.85? → Return THREAT
                ↓ No
        Return SAFE (with low confidence warning)
```

#### 3.2 Add Benign Classifier
Train a separate model to detect BENIGN inputs:

```python
class EnsembleWithBenignVeto:
    def predict(self, text):
        malicious_score = self.attack_model.predict(text)
        benign_score = self.benign_model.predict(text)
        
        # Benign model can veto attack classification
        if benign_score > 0.8 and malicious_score < 0.95:
            return {'is_attack': False, 'confidence': benign_score}
        
        return {'is_attack': malicious_score > 0.85, 'confidence': malicious_score}
```

#### 3.3 Context-Aware Features
Add features that help distinguish benign from malicious:

```python
def extract_context_features(text):
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
        'special_char_ratio': sum(1 for c in text if not c.isalnum()) / max(len(text), 1),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'has_sql_keywords': any(kw in text.upper() for kw in ['SELECT', 'DROP', 'INSERT', 'DELETE', 'UNION']),
        'has_script_tags': '<script' in text.lower(),
        'has_shell_chars': any(c in text for c in ['|', ';', '`', '$(']),
        'looks_like_sentence': text[0].isupper() and text[-1] in '.!?' if text else False,
    }
```

---

### Phase 4: Monitoring & Feedback Loop (Ongoing)

#### 4.1 Log False Positives
```python
def log_prediction(input_text, prediction, user_feedback=None):
    if user_feedback == 'false_positive':
        # Add to retraining queue
        save_to_fp_dataset(input_text)
```

#### 4.2 A/B Testing
Test new thresholds/models against production traffic.

#### 4.3 Weekly Retraining
Incorporate false positive feedback into training data.

---

## Implementation Priority

| Priority | Task | Impact | Effort |
|----------|------|--------|--------|
| P0 | Raise threshold to 0.85 | High | 5 min |
| P0 | Add benign pre-filter | High | 30 min |
| P1 | Length-based confidence scaling | Medium | 15 min |
| P1 | Temperature calibration | Medium | 1 hour |
| P2 | Retrain with balanced data | High | 4-8 hours |
| P2 | Add hard negatives | High | 2 hours |
| P3 | Two-stage classification | High | 4 hours |
| P3 | Benign classifier | Medium | 8 hours |

---

## Quick Win: Immediate Code Changes

### File: `src/batch_predictor.py`

Add this before the predict methods:

```python
class BenignPreFilter:
    SUSPICIOUS = set('<>;\'"\\|`${}()[]&')
    GREETINGS = {'hi', 'hey', 'hello', 'thanks', 'ok', 'yes', 'no', 'bye', 'please', 'sorry'}
    
    def is_benign(self, text: str) -> tuple[bool, float]:
        """Returns (is_benign, confidence)."""
        text = str(text).strip()
        lower = text.lower()
        
        # Common greetings
        if lower in self.GREETINGS:
            return True, 0.99
        
        # Short alphanumeric
        if len(text) < 20 and text.replace(' ', '').replace('.', '').replace(',', '').isalnum():
            return True, 0.95
        
        # No suspicious chars and reasonable length
        if len(text) < 50 and not any(c in self.SUSPICIOUS for c in text):
            if text[0].isupper() or lower.split()[0] in self.GREETINGS:
                return True, 0.90
        
        return False, 0.0
```

### File: `configs/optimal_thresholds.json`

```json
{
  "thresholds": {
    "default": 0.85,
    "payload_cnn": 0.85,
    "url_cnn": 0.80,
    ...
  }
}
```

---

## Expected Results After Phase 1

| Metric | Before | After |
|--------|--------|-------|
| False Positive Rate | 90%+ | <10% |
| "Hey" confidence | 95% threat | 0% (pre-filtered) |
| "O'Brien" confidence | 99.9% threat | <50% (scaled) |
| True Positive Rate | ~99% | ~95% (acceptable trade-off) |

The goal is to **prioritize precision over recall** for user-facing applications. Missing some attacks is better than flagging every normal message as malicious.
