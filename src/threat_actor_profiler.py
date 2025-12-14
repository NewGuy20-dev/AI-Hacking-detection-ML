"""Threat actor profiling and behavioral signature clustering."""
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans


class ThreatActorProfiler:
    """Profile and cluster threat actors by behavior."""
    
    def __init__(self):
        self.actors = defaultdict(lambda: {
            'attacks': [], 'targets': set(), 'protocols': set(),
            'times': [], 'techniques': defaultdict(int)
        })
        self.clusters = None
        self.cluster_labels = {}
    
    def record_attack(self, actor_ip: str, target_ip: str, attack_type: str,
                      protocol: str = 'TCP', hour: int = None):
        """Record an attack from a threat actor."""
        actor = self.actors[actor_ip]
        actor['attacks'].append(attack_type)
        actor['targets'].add(target_ip)
        actor['protocols'].add(protocol)
        actor['times'].append(hour or 12)
        actor['techniques'][attack_type] += 1
    
    def get_actor_features(self, actor_ip: str) -> np.ndarray:
        """Extract behavioral features for an actor."""
        actor = self.actors[actor_ip]
        if not actor['attacks']:
            return None
        
        return np.array([
            len(actor['attacks']),           # Total attacks
            len(actor['targets']),           # Unique targets
            len(actor['protocols']),         # Protocol diversity
            np.mean(actor['times']),         # Avg attack hour
            np.std(actor['times']) if len(actor['times']) > 1 else 0,  # Time variance
            len(actor['techniques']),        # Technique diversity
            max(actor['techniques'].values()),  # Most used technique count
        ])
    
    def cluster_actors(self, n_clusters: int = 3):
        """Cluster actors by behavioral similarity."""
        actors_with_data = [(ip, self.get_actor_features(ip)) 
                           for ip in self.actors if self.get_actor_features(ip) is not None]
        
        if len(actors_with_data) < n_clusters:
            return None
        
        ips, features = zip(*actors_with_data)
        X = np.array(features)
        
        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        self.cluster_labels = dict(zip(ips, labels))
        self.clusters = defaultdict(list)
        for ip, label in zip(ips, labels):
            self.clusters[label].append(ip)
        
        return {f'cluster_{i}': self.clusters[i] for i in range(n_clusters)}
    
    def get_profile(self, actor_ip: str) -> dict:
        """Get threat actor profile."""
        actor = self.actors[actor_ip]
        return {
            'ip': actor_ip,
            'total_attacks': len(actor['attacks']),
            'unique_targets': len(actor['targets']),
            'techniques': dict(actor['techniques']),
            'cluster': self.cluster_labels.get(actor_ip),
            'threat_level': self._calc_threat_level(actor)
        }
    
    def _calc_threat_level(self, actor: dict) -> str:
        score = len(actor['attacks']) + len(actor['targets']) * 2 + len(actor['techniques']) * 3
        if score > 20:
            return 'CRITICAL'
        elif score > 10:
            return 'HIGH'
        elif score > 5:
            return 'MEDIUM'
        return 'LOW'


if __name__ == '__main__':
    profiler = ThreatActorProfiler()
    
    # Simulate different actor behaviors
    # Actor 1: Aggressive scanner
    for i in range(10):
        profiler.record_attack('attacker.1', f'target.{i}', 'Probe', hour=3)
    
    # Actor 2: Targeted attacker
    for _ in range(5):
        profiler.record_attack('attacker.2', 'target.1', 'DoS', hour=14)
        profiler.record_attack('attacker.2', 'target.1', 'R2L', hour=15)
    
    # Actor 3: Opportunistic
    profiler.record_attack('attacker.3', 'target.5', 'Probe', hour=10)
    profiler.record_attack('attacker.3', 'target.6', 'DoS', hour=22)
    
    print("Threat Actor Profiles:")
    for ip in ['attacker.1', 'attacker.2', 'attacker.3']:
        print(f"  {profiler.get_profile(ip)}")
    
    clusters = profiler.cluster_actors(n_clusters=2)
    print(f"\nBehavioral Clusters: {clusters}")
