"""Attack correlation for linking related security events."""
from collections import defaultdict
from datetime import datetime, timedelta


class AttackCorrelator:
    """Correlate attacks across multiple sources and time windows."""
    
    def __init__(self, time_window_minutes: int = 15):
        self.events = []
        self.time_window = timedelta(minutes=time_window_minutes)
        self.correlations = []
    
    def add_event(self, event_type: str, source_ip: str, dest_ip: str = None,
                  attack_type: str = None, confidence: float = 0.5, metadata: dict = None):
        """Add a security event."""
        event = {
            'id': len(self.events),
            'timestamp': datetime.now(),
            'event_type': event_type,
            'source_ip': source_ip,
            'dest_ip': dest_ip,
            'attack_type': attack_type,
            'confidence': confidence,
            'metadata': metadata or {}
        }
        self.events.append(event)
        self._check_correlations(event)
        return event
    
    def _check_correlations(self, new_event):
        """Check if new event correlates with existing events."""
        cutoff = new_event['timestamp'] - self.time_window
        
        related = []
        for event in self.events[:-1]:  # Exclude the new event
            if event['timestamp'] < cutoff:
                continue
            
            # Correlation rules
            score = 0
            reasons = []
            
            # Same source IP
            if event['source_ip'] == new_event['source_ip']:
                score += 0.4
                reasons.append('same_source_ip')
            
            # Same destination
            if event['dest_ip'] and event['dest_ip'] == new_event['dest_ip']:
                score += 0.3
                reasons.append('same_dest_ip')
            
            # Related attack types (kill chain)
            if self._attacks_related(event['attack_type'], new_event['attack_type']):
                score += 0.3
                reasons.append('attack_chain')
            
            if score >= 0.4:
                related.append({
                    'event_id': event['id'],
                    'correlation_score': round(score, 2),
                    'reasons': reasons
                })
        
        if related:
            correlation = {
                'primary_event': new_event['id'],
                'timestamp': new_event['timestamp'].isoformat(),
                'related_events': related,
                'campaign_likelihood': min(1.0, sum(r['correlation_score'] for r in related))
            }
            self.correlations.append(correlation)
    
    def _attacks_related(self, type1: str, type2: str) -> bool:
        """Check if two attack types are part of a kill chain."""
        if not type1 or not type2:
            return False
        
        kill_chains = [
            {'Probe', 'DoS'},
            {'Probe', 'R2L'},
            {'R2L', 'U2R'},
            {'PortScan', 'exploit'},
            {'recon', 'exploit', 'persistence'}
        ]
        
        t1, t2 = type1.lower(), type2.lower()
        for chain in kill_chains:
            if any(t1 in c.lower() for c in chain) and any(t2 in c.lower() for c in chain):
                return True
        return False
    
    def get_campaigns(self) -> list:
        """Get identified attack campaigns."""
        campaigns = []
        for corr in self.correlations:
            if corr['campaign_likelihood'] >= 0.6:
                campaigns.append({
                    'events': [corr['primary_event']] + [r['event_id'] for r in corr['related_events']],
                    'likelihood': corr['campaign_likelihood'],
                    'timestamp': corr['timestamp']
                })
        return campaigns
    
    def summary(self) -> dict:
        return {
            'total_events': len(self.events),
            'correlations_found': len(self.correlations),
            'campaigns_identified': len(self.get_campaigns())
        }


if __name__ == '__main__':
    correlator = AttackCorrelator(time_window_minutes=30)
    
    # Simulate attack sequence
    correlator.add_event('network', '192.168.1.100', '10.0.0.1', 'Probe', 0.8)
    correlator.add_event('network', '192.168.1.100', '10.0.0.2', 'Probe', 0.75)
    correlator.add_event('network', '192.168.1.100', '10.0.0.1', 'DoS', 0.9)
    correlator.add_event('network', '192.168.1.101', '10.0.0.3', 'Probe', 0.6)  # Different attacker
    
    print("Attack Correlation Results:")
    print(f"  Summary: {correlator.summary()}")
    
    campaigns = correlator.get_campaigns()
    if campaigns:
        print(f"  Campaigns detected: {len(campaigns)}")
        for c in campaigns:
            print(f"    - Events {c['events']}, likelihood={c['likelihood']:.2f}")
