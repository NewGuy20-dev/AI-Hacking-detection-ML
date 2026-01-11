"""Alert management system for security detections with explainability."""
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Dict, Optional, Any

try:
    from explainer import Explainer, explain_prediction
    from checklist import ChecklistGenerator, generate_checklist
except ImportError:
    Explainer = None
    explain_prediction = None
    ChecklistGenerator = None
    generate_checklist = None


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertManager:
    """Generate and manage security alerts with explainability."""
    
    def __init__(self, output_dir: str = None):
        base = Path(__file__).parent.parent
        self.output_dir = Path(output_dir) if output_dir else base / 'alerts'
        self.output_dir.mkdir(exist_ok=True)
        self.alerts = []
        self.explainer = Explainer() if Explainer else None
        self.checklist_gen = ChecklistGenerator() if ChecklistGenerator else None
    
    def create_alert(self, attack_type: str, confidence: float, source_ip: str = None,
                     dest_ip: str = None, details: dict = None, 
                     payload: str = None, model_scores: Dict[str, float] = None,
                     include_explanation: bool = True,
                     include_checklist: bool = True) -> dict:
        """Create a structured alert with optional explainability."""
        severity = self._calculate_severity(confidence, attack_type)
        
        alert = {
            'id': f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
            'timestamp': datetime.now().isoformat(),
            'severity': severity.name,
            'severity_level': severity.value,
            'attack_type': attack_type,
            'confidence': round(confidence, 4),
            'source_ip': source_ip,
            'dest_ip': dest_ip,
            'details': details or {},
            'model_scores': model_scores or {},
            'recommended_actions': self._get_recommendations(severity, attack_type),
            'status': 'NEW'
        }
        
        # Add explanation if available
        if include_explanation and self.explainer and payload:
            explanation = self._generate_explanation(payload, confidence, attack_type, model_scores)
            alert['explanation'] = explanation
            alert['verdict'] = explanation.get('verdict', 'UNKNOWN')
            alert['indicators'] = explanation.get('indicators', {})
        
        # Add analyst checklist if available
        if include_checklist and self.checklist_gen:
            checklist = self.checklist_gen.generate(
                alert['id'], attack_type, severity.name.lower(), confidence
            )
            alert['checklist'] = checklist.to_dict()
        
        self.alerts.append(alert)
        return alert
    
    def _generate_explanation(self, payload: str, confidence: float, 
                             attack_type: str, model_scores: Dict[str, float] = None) -> Dict:
        """Generate explanation for the alert."""
        if not self.explainer:
            return {}
        
        # Determine data type from attack type
        if attack_type in ['malicious_url', 'phishing']:
            data_type = 'url'
        elif attack_type in ['network_attack', 'DoS', 'DDoS', 'Probe']:
            data_type = 'network'
        else:
            data_type = 'payload'
        
        try:
            if data_type == 'payload':
                explanation = self.explainer.explain_payload(payload, confidence, model_scores)
            elif data_type == 'url':
                explanation = self.explainer.explain_url(payload, confidence, model_scores)
            else:
                explanation = self.explainer.explain_ensemble(
                    payload, model_scores or {}, confidence, data_type
                )
            return explanation.to_dict()
        except Exception:
            return {}
    
    def _calculate_severity(self, confidence: float, attack_type: str) -> Severity:
        critical_types = {'U2R', 'rootkit', 'buffer_overflow', 'sqlattack', 'command_injection'}
        high_types = {'R2L', 'DoS', 'DDoS', 'backdoor', 'sql_injection', 'xss'}
        
        if attack_type.lower() in critical_types or confidence > 0.95:
            return Severity.CRITICAL
        elif attack_type.lower() in high_types or confidence > 0.85:
            return Severity.HIGH
        elif confidence > 0.7:
            return Severity.MEDIUM
        return Severity.LOW
    
    def _get_recommendations(self, severity: Severity, attack_type: str) -> list:
        base_actions = ['Log event for analysis']
        
        if severity == Severity.LOW:
            return base_actions + ['Monitor for repeated activity']
        elif severity == Severity.MEDIUM:
            return base_actions + ['Increase logging verbosity', 'Review related traffic']
        elif severity == Severity.HIGH:
            return base_actions + ['Consider blocking source IP', 'Alert security team', 'Capture forensic data']
        else:  # CRITICAL
            return base_actions + ['IMMEDIATE: Block source IP', 'Isolate affected host', 
                                   'Escalate to incident response', 'Preserve all evidence']
    
    def create_alert_with_explanation(self, payload: str, confidence: float,
                                      attack_type: str = None, source_ip: str = None,
                                      model_scores: Dict[str, float] = None) -> dict:
        """Create alert with full explanation (convenience method)."""
        # Auto-detect attack type if not provided
        if not attack_type and self.explainer:
            attack_type = self.explainer.detect_attack_type(payload) or "unknown"
        
        return self.create_alert(
            attack_type=attack_type or "unknown",
            confidence=confidence,
            source_ip=source_ip,
            payload=payload,
            model_scores=model_scores,
            include_explanation=True,
            include_checklist=True
        )
    
    def export_json(self, filepath: str = None):
        """Export alerts to JSON."""
        path = filepath or self.output_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
        with open(path, 'w') as f:
            json.dump(self.alerts, f, indent=2)
        return str(path)
    
    def export_syslog(self) -> list:
        """Format alerts for syslog."""
        lines = []
        for alert in self.alerts:
            line = f"<{alert['severity_level'] + 10}>{alert['timestamp']} HACKING-DETECT {alert['id']}: " \
                   f"type={alert['attack_type']} conf={alert['confidence']} src={alert['source_ip']} " \
                   f"dst={alert.get('dest_ip')} severity={alert['severity']}"
            lines.append(line)
        return lines
    
    def summary(self) -> dict:
        """Get alert summary."""
        by_severity = {}
        by_verdict = {}
        for alert in self.alerts:
            sev = alert['severity']
            by_severity[sev] = by_severity.get(sev, 0) + 1
            verdict = alert.get('verdict', 'UNKNOWN')
            by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
        return {
            'total': len(self.alerts), 
            'by_severity': by_severity,
            'by_verdict': by_verdict
        }


if __name__ == '__main__':
    am = AlertManager()
    
    # Test alerts
    am.create_alert('DoS', 0.92, '192.168.1.100', '10.0.0.1', {'packets': 50000})
    am.create_alert('Probe', 0.75, '192.168.1.101', '10.0.0.2')
    am.create_alert('U2R', 0.98, '192.168.1.102', '10.0.0.3', {'exploit': 'buffer_overflow'})
    
    print("Generated Alerts:")
    for alert in am.alerts:
        print(f"  [{alert['severity']:8s}] {alert['attack_type']:10s} conf={alert['confidence']:.2f}")
    
    print(f"\nSummary: {am.summary()}")
    print(f"Exported to: {am.export_json()}")
