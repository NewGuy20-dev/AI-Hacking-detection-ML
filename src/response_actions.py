"""Automated response actions for detected threats."""
from datetime import datetime
from pathlib import Path


class ResponseActions:
    """Generate response recommendations and commands."""
    
    def __init__(self):
        self.action_log = []
    
    def recommend(self, severity: str, source_ip: str = None, attack_type: str = None) -> dict:
        """Generate response recommendations based on severity."""
        actions = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'source_ip': source_ip,
            'attack_type': attack_type,
            'recommendations': [],
            'commands': []
        }
        
        if severity == 'LOW':
            actions['recommendations'] = ['Continue monitoring', 'Log for baseline']
        
        elif severity == 'MEDIUM':
            actions['recommendations'] = ['Increase logging', 'Rate limit source', 'Review in 1 hour']
            if source_ip:
                actions['commands'].append(f"# Rate limit\niptables -A INPUT -s {source_ip} -m limit --limit 10/min -j ACCEPT")
        
        elif severity == 'HIGH':
            actions['recommendations'] = ['Block source IP', 'Alert SOC', 'Capture traffic']
            if source_ip:
                actions['commands'].extend([
                    f"# Block IP\niptables -A INPUT -s {source_ip} -j DROP",
                    f"# Capture traffic\ntcpdump -i any host {source_ip} -w /tmp/capture_{source_ip}.pcap &"
                ])
        
        elif severity == 'CRITICAL':
            actions['recommendations'] = ['IMMEDIATE: Isolate host', 'Block at firewall', 
                                          'Escalate to IR team', 'Preserve evidence']
            if source_ip:
                actions['commands'].extend([
                    f"# Emergency block\niptables -I INPUT 1 -s {source_ip} -j DROP",
                    f"iptables -I OUTPUT 1 -d {source_ip} -j DROP",
                    f"# Log all connections\nnetstat -an | grep {source_ip} >> /var/log/incident_{source_ip}.log"
                ])
        
        self.action_log.append(actions)
        return actions
    
    def generate_firewall_rules(self, blocked_ips: list) -> str:
        """Generate iptables rules for blocked IPs."""
        rules = ["#!/bin/bash", "# Auto-generated firewall rules", ""]
        for ip in blocked_ips:
            rules.append(f"iptables -A INPUT -s {ip} -j DROP")
            rules.append(f"iptables -A OUTPUT -d {ip} -j DROP")
        return "\n".join(rules)
    
    def export_log(self, filepath: str = None) -> str:
        import json
        base = Path('/workspaces/AI-Hacking-detection-ML')
        path = filepath or base / 'alerts' / 'response_log.json'
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.action_log, f, indent=2)
        return str(path)


if __name__ == '__main__':
    ra = ResponseActions()
    
    print("Response Recommendations:")
    for sev in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']:
        result = ra.recommend(sev, '192.168.1.100', 'DoS')
        print(f"\n[{sev}]")
        print(f"  Actions: {result['recommendations']}")
        if result['commands']:
            print(f"  Commands: {len(result['commands'])} generated")
