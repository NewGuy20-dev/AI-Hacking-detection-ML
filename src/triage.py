"""Fast triage system for alert prioritization and quick verdicts."""
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from enum import Enum


class Priority(Enum):
    """Alert priority levels."""
    P1_CRITICAL = 1  # Immediate action required
    P2_HIGH = 2      # Investigate within 1 hour
    P3_MEDIUM = 3    # Investigate within 4 hours
    P4_LOW = 4       # Review within 24 hours
    P5_INFO = 5      # Batch review


class Verdict(Enum):
    """Quick verdict classifications."""
    MALICIOUS = "MALICIOUS"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_BENIGN = "LIKELY_BENIGN"
    BENIGN = "BENIGN"


@dataclass
class TriageResult:
    """Result of triage analysis."""
    alert_id: str
    priority: Priority
    verdict: Verdict
    confidence: float
    attack_type: Optional[str]
    sla_hours: int
    auto_action: Optional[str]
    review_required: bool
    factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "priority": self.priority.name,
            "priority_level": self.priority.value,
            "verdict": self.verdict.value,
            "confidence": round(self.confidence, 4),
            "attack_type": self.attack_type,
            "sla_hours": self.sla_hours,
            "auto_action": self.auto_action,
            "review_required": self.review_required,
            "factors": self.factors
        }


class TriageEngine:
    """Fast triage engine for alert prioritization."""
    
    # SLA hours per priority
    SLA_HOURS = {
        Priority.P1_CRITICAL: 0,  # Immediate
        Priority.P2_HIGH: 1,
        Priority.P3_MEDIUM: 4,
        Priority.P4_LOW: 24,
        Priority.P5_INFO: 48,
    }
    
    # High-risk attack types
    CRITICAL_ATTACKS = {'command_injection', 'rootkit', 'buffer_overflow', 'U2R'}
    HIGH_ATTACKS = {'sql_injection', 'xss', 'R2L', 'DoS', 'backdoor'}
    
    # Confidence thresholds
    THRESHOLDS = {
        'critical': 0.95,
        'high': 0.80,
        'medium': 0.60,
        'low': 0.40,
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        if config_path and config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                self.THRESHOLDS = config.get('priority_thresholds', self.THRESHOLDS)
    
    def calculate_priority(self, confidence: float, attack_type: str,
                          source_reputation: float = 0.5) -> Priority:
        """Calculate alert priority based on multiple factors."""
        # Start with confidence-based priority
        if confidence >= self.THRESHOLDS.get('P1_critical', 0.95):
            priority = Priority.P1_CRITICAL
        elif confidence >= self.THRESHOLDS.get('P2_high', 0.80):
            priority = Priority.P2_HIGH
        elif confidence >= self.THRESHOLDS.get('P3_medium', 0.60):
            priority = Priority.P3_MEDIUM
        elif confidence >= self.THRESHOLDS.get('P4_low', 0.40):
            priority = Priority.P4_LOW
        else:
            priority = Priority.P5_INFO
        
        # Escalate for critical attack types
        if attack_type in self.CRITICAL_ATTACKS:
            if priority.value > Priority.P1_CRITICAL.value:
                priority = Priority(max(1, priority.value - 1))
        elif attack_type in self.HIGH_ATTACKS:
            if priority.value > Priority.P2_HIGH.value:
                priority = Priority(max(2, priority.value - 1))
        
        # Escalate for bad source reputation
        if source_reputation > 0.8:  # Known bad source
            if priority.value > Priority.P2_HIGH.value:
                priority = Priority(max(2, priority.value - 1))
        
        return priority
    
    def get_verdict(self, confidence: float) -> Verdict:
        """Determine quick verdict based on confidence."""
        if confidence >= 0.90:
            return Verdict.MALICIOUS
        elif confidence >= 0.60:
            return Verdict.SUSPICIOUS
        elif confidence >= 0.30:
            return Verdict.LIKELY_BENIGN
        return Verdict.BENIGN
    
    def get_auto_action(self, priority: Priority, verdict: Verdict,
                       attack_type: str) -> Optional[str]:
        """Determine automatic action if applicable."""
        if priority == Priority.P1_CRITICAL and verdict == Verdict.MALICIOUS:
            if attack_type in self.CRITICAL_ATTACKS:
                return "AUTO_BLOCK_AND_ISOLATE"
            return "AUTO_BLOCK_SOURCE"
        
        if priority == Priority.P2_HIGH and verdict == Verdict.MALICIOUS:
            return "AUTO_BLOCK_SOURCE"
        
        if verdict == Verdict.BENIGN and priority == Priority.P5_INFO:
            return "AUTO_CLOSE"
        
        return None
    
    def triage(self, alert_id: str, confidence: float, attack_type: str,
              source_reputation: float = 0.5, model_scores: Dict[str, float] = None) -> TriageResult:
        """Perform full triage on an alert."""
        priority = self.calculate_priority(confidence, attack_type, source_reputation)
        verdict = self.get_verdict(confidence)
        auto_action = self.get_auto_action(priority, verdict, attack_type)
        
        # Determine if human review is required
        review_required = (
            verdict in [Verdict.SUSPICIOUS, Verdict.LIKELY_BENIGN] or
            priority in [Priority.P1_CRITICAL, Priority.P2_HIGH] or
            auto_action is None
        )
        
        # Collect factors that influenced the decision
        factors = []
        if confidence >= 0.95:
            factors.append(f"High confidence: {confidence:.1%}")
        if attack_type in self.CRITICAL_ATTACKS:
            factors.append(f"Critical attack type: {attack_type}")
        elif attack_type in self.HIGH_ATTACKS:
            factors.append(f"High-risk attack type: {attack_type}")
        if source_reputation > 0.8:
            factors.append(f"Bad source reputation: {source_reputation:.1%}")
        if model_scores:
            high_scores = [f"{k}: {v:.1%}" for k, v in model_scores.items() if v > 0.7]
            if high_scores:
                factors.append(f"Model alerts: {', '.join(high_scores)}")
        
        return TriageResult(
            alert_id=alert_id,
            priority=priority,
            verdict=verdict,
            confidence=confidence,
            attack_type=attack_type,
            sla_hours=self.SLA_HOURS[priority],
            auto_action=auto_action,
            review_required=review_required,
            factors=factors
        )
    
    def batch_triage(self, alerts: List[Dict]) -> List[TriageResult]:
        """Triage multiple alerts efficiently."""
        results = []
        for alert in alerts:
            result = self.triage(
                alert_id=alert.get('id', 'unknown'),
                confidence=alert.get('confidence', 0.5),
                attack_type=alert.get('attack_type', 'unknown'),
                source_reputation=alert.get('source_reputation', 0.5),
                model_scores=alert.get('model_scores')
            )
            results.append(result)
        
        # Sort by priority (P1 first)
        results.sort(key=lambda x: x.priority.value)
        return results
    
    def get_queue_summary(self, results: List[TriageResult]) -> Dict:
        """Get summary of triage queue."""
        by_priority = {}
        by_verdict = {}
        auto_actions = 0
        review_needed = 0
        
        for r in results:
            by_priority[r.priority.name] = by_priority.get(r.priority.name, 0) + 1
            by_verdict[r.verdict.value] = by_verdict.get(r.verdict.value, 0) + 1
            if r.auto_action:
                auto_actions += 1
            if r.review_required:
                review_needed += 1
        
        return {
            "total": len(results),
            "by_priority": by_priority,
            "by_verdict": by_verdict,
            "auto_actions": auto_actions,
            "review_needed": review_needed,
            "immediate_attention": by_priority.get("P1_CRITICAL", 0) + by_priority.get("P2_HIGH", 0)
        }


def triage_alert(alert_id: str, confidence: float, attack_type: str,
                source_reputation: float = 0.5) -> Dict:
    """Convenience function for single alert triage."""
    engine = TriageEngine()
    result = engine.triage(alert_id, confidence, attack_type, source_reputation)
    return result.to_dict()


def batch_triage_alerts(alerts: List[Dict]) -> List[Dict]:
    """Convenience function for batch triage."""
    engine = TriageEngine()
    results = engine.batch_triage(alerts)
    return [r.to_dict() for r in results]
