"""Unified explanation engine for model predictions."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from indicators import (
    IndicatorSet, Indicator, get_indicator_extractor,
    PayloadIndicatorExtractor, URLIndicatorExtractor, NetworkIndicatorExtractor
)

try:
    from context_classifier import ContextAwareClassifier
    HAS_CONTEXT = True
except ImportError:
    HAS_CONTEXT = False
    ContextAwareClassifier = None


@dataclass
class Explanation:
    """Complete explanation for a detection."""
    verdict: str  # MALICIOUS, SUSPICIOUS, LIKELY_BENIGN, BENIGN
    confidence: float
    attack_type: Optional[str]
    indicators: IndicatorSet
    model_scores: Dict[str, float]
    summary: str
    recommended_actions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "attack_type": self.attack_type,
            "indicators": self.indicators.to_dict(),
            "model_scores": {k: round(v, 4) for k, v in self.model_scores.items()},
            "summary": self.summary,
            "recommended_actions": self.recommended_actions
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class Explainer:
    """Unified explanation engine."""
    
    ATTACK_TYPES = {
        'sql_injection': ['SELECT', 'UNION', 'INSERT', '--', "' OR"],
        'xss': ['<script', 'javascript:', 'onerror', 'onload'],
        'command_injection': ['|', ';', '`', '$(', '&&'],
        'path_traversal': ['../', '..\\', '%2e%2e'],
        'malicious_url': ['phishing', 'malware'],
        'network_attack': ['scan', 'flood', 'dos'],
    }
    
    def __init__(self, use_context: bool = True):
        self.payload_extractor = PayloadIndicatorExtractor()
        self.url_extractor = URLIndicatorExtractor()
        self.network_extractor = NetworkIndicatorExtractor()
        self.context_classifier = ContextAwareClassifier() if (use_context and HAS_CONTEXT) else None
    
    def get_verdict(self, confidence: float, text: str = None) -> str:
        """Determine verdict based on confidence, with optional context adjustment."""
        # Apply context adjustment if available
        if self.context_classifier and text:
            result = self.context_classifier.adjust_score(text, confidence)
            confidence = result.adjusted_score
        
        if confidence >= 0.90:
            return "MALICIOUS"
        elif confidence >= 0.60:
            return "SUSPICIOUS"
        elif confidence >= 0.30:
            return "LIKELY_BENIGN"
        return "BENIGN"
    
    def detect_attack_type(self, text: str) -> Optional[str]:
        """Detect specific attack type from content."""
        text_lower = text.lower()
        
        for attack_type, patterns in self.ATTACK_TYPES.items():
            matches = sum(1 for p in patterns if p.lower() in text_lower)
            if matches >= 2:
                return attack_type
        
        return None
    
    def generate_summary(self, verdict: str, attack_type: Optional[str], 
                        indicators: IndicatorSet) -> str:
        """Generate human-readable summary."""
        if verdict == "BENIGN":
            return "No malicious indicators detected. Content appears benign."
        
        if verdict == "LIKELY_BENIGN":
            return "Low confidence detection. May be a false positive. Review recommended."
        
        primary_count = len(indicators.primary)
        attack_str = f" ({attack_type.replace('_', ' ')})" if attack_type else ""
        
        if verdict == "MALICIOUS":
            return f"High confidence malicious detection{attack_str}. {primary_count} primary indicators found. Immediate action recommended."
        
        return f"Suspicious activity detected{attack_str}. {primary_count} indicators require review."
    
    def get_recommended_actions(self, verdict: str, attack_type: Optional[str]) -> List[str]:
        """Get recommended actions based on verdict and attack type."""
        if verdict == "BENIGN":
            return ["No action required", "Auto-close candidate"]
        
        if verdict == "LIKELY_BENIGN":
            return ["Spot check for false positive", "Review source context"]
        
        actions = ["Review alert details", "Check source IP/user"]
        
        if attack_type == "sql_injection":
            actions.extend(["Check database logs", "Verify input sanitization"])
        elif attack_type == "xss":
            actions.extend(["Check for stored XSS", "Review output encoding"])
        elif attack_type == "command_injection":
            actions.extend(["Check system logs", "Review process execution"])
        elif attack_type == "malicious_url":
            actions.extend(["Block URL at proxy", "Check for user clicks"])
        
        if verdict == "MALICIOUS":
            actions.insert(0, "Block source immediately")
            actions.append("Escalate to incident response")
        
        return actions
    
    def explain_payload(self, text: str, score: float, 
                       model_scores: Optional[Dict[str, float]] = None) -> Explanation:
        """Generate explanation for payload detection."""
        indicators = self.payload_extractor.extract(text, score)
        verdict = self.get_verdict(score)
        attack_type = self.detect_attack_type(text)
        
        return Explanation(
            verdict=verdict,
            confidence=score,
            attack_type=attack_type,
            indicators=indicators,
            model_scores=model_scores or {"payload": score},
            summary=self.generate_summary(verdict, attack_type, indicators),
            recommended_actions=self.get_recommended_actions(verdict, attack_type)
        )
    
    def explain_url(self, url: str, score: float,
                   model_scores: Optional[Dict[str, float]] = None) -> Explanation:
        """Generate explanation for URL detection."""
        indicators = self.url_extractor.extract(url, score)
        verdict = self.get_verdict(score)
        
        return Explanation(
            verdict=verdict,
            confidence=score,
            attack_type="malicious_url" if verdict in ["MALICIOUS", "SUSPICIOUS"] else None,
            indicators=indicators,
            model_scores=model_scores or {"url": score},
            summary=self.generate_summary(verdict, "malicious_url" if score > 0.6 else None, indicators),
            recommended_actions=self.get_recommended_actions(verdict, "malicious_url")
        )
    
    def explain_network(self, flow_data: Dict, score: float,
                       model_scores: Optional[Dict[str, float]] = None) -> Explanation:
        """Generate explanation for network detection."""
        indicators = self.network_extractor.extract(flow_data, score)
        verdict = self.get_verdict(score)
        
        return Explanation(
            verdict=verdict,
            confidence=score,
            attack_type="network_attack" if verdict in ["MALICIOUS", "SUSPICIOUS"] else None,
            indicators=indicators,
            model_scores=model_scores or {"network": score},
            summary=self.generate_summary(verdict, "network_attack" if score > 0.6 else None, indicators),
            recommended_actions=self.get_recommended_actions(verdict, "network_attack")
        )
    
    def explain_ensemble(self, data: Any, model_scores: Dict[str, float], 
                        ensemble_score: float, data_type: str = "payload") -> Explanation:
        """Generate explanation for ensemble detection."""
        # Get indicators based on data type
        if data_type == "payload":
            indicators = self.payload_extractor.extract(str(data), ensemble_score)
            attack_type = self.detect_attack_type(str(data))
        elif data_type == "url":
            indicators = self.url_extractor.extract(str(data), ensemble_score)
            attack_type = "malicious_url" if ensemble_score > 0.6 else None
        else:
            indicators = IndicatorSet()
            attack_type = None
        
        verdict = self.get_verdict(ensemble_score)
        
        # Add model contribution indicators
        for model, score in model_scores.items():
            if score > 0.5:
                indicators.secondary.append(Indicator(
                    name=f"{model.title()} Detection",
                    description=f"{model} model flagged with {score:.1%} confidence",
                    severity="medium" if score > 0.7 else "low",
                    evidence=f"score={score:.4f}",
                    confidence=score
                ))
        
        return Explanation(
            verdict=verdict,
            confidence=ensemble_score,
            attack_type=attack_type,
            indicators=indicators,
            model_scores=model_scores,
            summary=self.generate_summary(verdict, attack_type, indicators),
            recommended_actions=self.get_recommended_actions(verdict, attack_type)
        )


def explain_prediction(data: Any, score: float, data_type: str = "payload",
                      model_scores: Optional[Dict[str, float]] = None) -> Dict:
    """Convenience function to explain a prediction."""
    explainer = Explainer()
    
    if data_type == "payload":
        explanation = explainer.explain_payload(str(data), score, model_scores)
    elif data_type == "url":
        explanation = explainer.explain_url(str(data), score, model_scores)
    elif data_type == "network":
        explanation = explainer.explain_network(data, score, model_scores)
    else:
        explanation = explainer.explain_ensemble(data, model_scores or {}, score, data_type)
    
    return explanation.to_dict()
