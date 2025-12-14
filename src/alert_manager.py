"""Alert management system for security detections."""
import json
from datetime import datetime
from pathlib import Path
from enum import Enum


class Severity(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class AlertManager:
    """Generate and manage security alerts."""
    
    def __init__(self, output_dir: str = None):
        base = Path('/workspaces/AI-Hacking-detection-ML')
        self.output_dir = Path(output_dir) if output_dir else base / 'alerts'
        self.output_dir.mkdir(exist_ok=True)
        self.alerts = []
    
    def create_alert(self, attack_type: str, confidence: float, source_ip: str = None,
                     dest_ip: str = None, details: dict = None) -> dict:
        """Create a structured alert."""
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
            'recommended_actions': self._get_recommendations(severity, attack_type),
            'status': 'NEW'
        }
        
        self.alerts.append(alert)
        return alert
    
    def _calculate_severity(self, confidence: float, attack_type: str) -> Severity:
        critical_types = {'U2R', 'rootkit', 'buffer_overflow', 'sqlattack'}
        high_types = {'R2L', 'DoS', 'DDoS', 'backdoor'}
        
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
                   f"dst={alert['dest_ip']} severity={alert['severity']}"
            lines.append(line)
        return lines
    
    def summary(self) -> dict:
        """Get alert summary."""
        by_severity = {}
        for alert in self.alerts:
            sev = alert['severity']
            by_severity[sev] = by_severity.get(sev, 0) + 1
        return {'total': len(self.alerts), 'by_severity': by_severity}


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
