"""Forensic logging for incident investigation."""
import json
import gzip
from datetime import datetime
from pathlib import Path


class ForensicLogger:
    """Capture and store forensic evidence."""
    
    def __init__(self, output_dir: str = None):
        base = Path('/workspaces/AI-Hacking-detection-ML')
        self.output_dir = Path(output_dir) if output_dir else base / 'forensics'
        self.output_dir.mkdir(exist_ok=True)
        self.current_incident = None
        self.events = []
    
    def start_incident(self, incident_id: str = None, description: str = None) -> str:
        """Start a new incident for logging."""
        self.current_incident = incident_id or f"INC-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.events = [{
            'type': 'INCIDENT_START',
            'timestamp': datetime.now().isoformat(),
            'incident_id': self.current_incident,
            'description': description
        }]
        return self.current_incident
    
    def log_event(self, event_type: str, data: dict):
        """Log a forensic event."""
        event = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'incident_id': self.current_incident,
            'data': data
        }
        self.events.append(event)
        return event
    
    def log_detection(self, attack_type: str, confidence: float, features: dict = None):
        """Log a detection event."""
        return self.log_event('DETECTION', {
            'attack_type': attack_type,
            'confidence': confidence,
            'features': features
        })
    
    def log_network(self, src_ip: str, dst_ip: str, protocol: str, payload_size: int = None):
        """Log network activity."""
        return self.log_event('NETWORK', {
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'protocol': protocol,
            'payload_size': payload_size
        })
    
    def log_response(self, action: str, result: str):
        """Log response action taken."""
        return self.log_event('RESPONSE', {
            'action': action,
            'result': result
        })
    
    def close_incident(self, resolution: str = None):
        """Close the current incident."""
        self.log_event('INCIDENT_CLOSE', {'resolution': resolution})
        self._save_incident()
        incident_id = self.current_incident
        self.current_incident = None
        self.events = []
        return incident_id
    
    def _save_incident(self):
        """Save incident to compressed file."""
        if not self.current_incident:
            return
        
        filepath = self.output_dir / f"{self.current_incident}.json.gz"
        with gzip.open(filepath, 'wt') as f:
            json.dump({
                'incident_id': self.current_incident,
                'event_count': len(self.events),
                'events': self.events
            }, f, indent=2)
        return str(filepath)
    
    def get_timeline(self) -> list:
        """Get chronological event timeline."""
        return sorted(self.events, key=lambda x: x['timestamp'])


if __name__ == '__main__':
    fl = ForensicLogger()
    
    # Simulate incident
    inc_id = fl.start_incident(description="Suspected DoS attack")
    print(f"Started incident: {inc_id}")
    
    fl.log_detection('DoS', 0.92, {'packets_per_sec': 50000})
    fl.log_network('192.168.1.100', '10.0.0.1', 'TCP', 1500)
    fl.log_response('block_ip', 'success')
    
    fl.close_incident('Blocked attacker IP, attack mitigated')
    
    print(f"Incident closed. Events logged: {len(fl.get_timeline())}")
    print(f"Saved to: forensics/{inc_id}.json.gz")
