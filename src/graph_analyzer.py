"""Graph-based network topology and attack path analysis."""
from collections import defaultdict
import json


class NetworkGraph:
    """Model network topology and identify attack paths."""
    
    def __init__(self):
        self.nodes = {}  # ip -> {type, first_seen, connections}
        self.edges = defaultdict(lambda: {'count': 0, 'protocols': set(), 'attacks': []})
    
    def add_connection(self, src_ip: str, dst_ip: str, protocol: str = 'TCP', 
                       is_attack: bool = False, attack_type: str = None):
        """Add a network connection."""
        for ip in [src_ip, dst_ip]:
            if ip not in self.nodes:
                self.nodes[ip] = {'connections': 0, 'as_source': 0, 'as_dest': 0}
        
        self.nodes[src_ip]['as_source'] += 1
        self.nodes[dst_ip]['as_dest'] += 1
        self.nodes[src_ip]['connections'] += 1
        self.nodes[dst_ip]['connections'] += 1
        
        edge_key = (src_ip, dst_ip)
        self.edges[edge_key]['count'] += 1
        self.edges[edge_key]['protocols'].add(protocol)
        if is_attack:
            self.edges[edge_key]['attacks'].append(attack_type)
    
    def find_attack_paths(self, target_ip: str, max_depth: int = 3) -> list:
        """Find all paths leading to attacks on target."""
        paths = []
        for (src, dst), data in self.edges.items():
            if dst == target_ip and data['attacks']:
                path = {'source': src, 'target': dst, 'attacks': data['attacks'], 
                        'connection_count': data['count']}
                # Check for lateral movement (src was also a target)
                for (s2, d2), d2_data in self.edges.items():
                    if d2 == src and d2_data['attacks']:
                        path['lateral_from'] = s2
                paths.append(path)
        return paths
    
    def get_high_risk_nodes(self, threshold: int = 10) -> list:
        """Identify nodes with suspicious connection patterns."""
        risky = []
        for ip, data in self.nodes.items():
            # High outbound = potential scanner/attacker
            if data['as_source'] > threshold and data['as_source'] > data['as_dest'] * 3:
                risky.append({'ip': ip, 'risk': 'scanner', 'outbound': data['as_source']})
            # High inbound = potential target
            elif data['as_dest'] > threshold:
                risky.append({'ip': ip, 'risk': 'target', 'inbound': data['as_dest']})
        return sorted(risky, key=lambda x: x.get('outbound', 0) + x.get('inbound', 0), reverse=True)
    
    def summary(self) -> dict:
        return {'nodes': len(self.nodes), 'edges': len(self.edges), 
                'attack_edges': sum(1 for e in self.edges.values() if e['attacks'])}


if __name__ == '__main__':
    g = NetworkGraph()
    # Simulate attack scenario
    g.add_connection('attacker.1', 'target.1', 'TCP', True, 'Probe')
    g.add_connection('attacker.1', 'target.2', 'TCP', True, 'Probe')
    g.add_connection('attacker.1', 'target.1', 'TCP', True, 'DoS')
    g.add_connection('normal.1', 'server.1', 'TCP')
    
    print(f"Graph: {g.summary()}")
    print(f"Attack paths to target.1: {g.find_attack_paths('target.1')}")
    print(f"High risk nodes: {g.get_high_risk_nodes(threshold=1)}")
