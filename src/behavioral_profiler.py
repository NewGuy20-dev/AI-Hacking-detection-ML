"""Behavioral profiling for anomaly detection based on entity baselines."""
import numpy as np
from collections import defaultdict
from datetime import datetime


class BehavioralProfiler:
    """Track and analyze entity behavior patterns."""
    
    def __init__(self, window_size: int = 100):
        self.profiles = defaultdict(lambda: {
            'observations': [],
            'mean': None,
            'std': None,
            'last_updated': None
        })
        self.window_size = window_size
    
    def update(self, entity_id: str, features: np.ndarray):
        """Update profile with new observation."""
        profile = self.profiles[entity_id]
        profile['observations'].append(features)
        
        # Keep only recent observations
        if len(profile['observations']) > self.window_size:
            profile['observations'] = profile['observations'][-self.window_size:]
        
        # Update statistics
        obs = np.array(profile['observations'])
        profile['mean'] = obs.mean(axis=0)
        profile['std'] = obs.std(axis=0) + 1e-8  # Avoid division by zero
        profile['last_updated'] = datetime.now().isoformat()
    
    def score(self, entity_id: str, features: np.ndarray) -> dict:
        """Score how much current behavior deviates from baseline."""
        profile = self.profiles[entity_id]
        
        if profile['mean'] is None or len(profile['observations']) < 5:
            return {'deviation_score': 0.0, 'is_anomalous': False, 'reason': 'insufficient_data'}
        
        # Z-score based deviation
        z_scores = np.abs((features - profile['mean']) / profile['std'])
        max_z = float(z_scores.max())
        mean_z = float(z_scores.mean())
        
        # Anomaly if any feature > 3 std or mean > 2 std
        is_anomalous = max_z > 3.0 or mean_z > 2.0
        
        return {
            'deviation_score': round(mean_z, 3),
            'max_deviation': round(max_z, 3),
            'is_anomalous': is_anomalous,
            'observations_count': len(profile['observations'])
        }
    
    def get_profile(self, entity_id: str) -> dict:
        """Get profile summary for an entity."""
        profile = self.profiles[entity_id]
        if profile['mean'] is None:
            return {'entity_id': entity_id, 'status': 'no_data'}
        
        return {
            'entity_id': entity_id,
            'observations': len(profile['observations']),
            'mean': profile['mean'].tolist() if profile['mean'] is not None else None,
            'last_updated': profile['last_updated']
        }


if __name__ == '__main__':
    profiler = BehavioralProfiler(window_size=20)
    
    # Simulate normal behavior
    entity = 'user_192.168.1.100'
    print(f"Building profile for {entity}...")
    
    for i in range(15):
        normal = np.array([100 + np.random.randn()*5, 50 + np.random.randn()*3, 10 + np.random.randn()])
        profiler.update(entity, normal)
    
    # Test normal behavior
    normal_test = np.array([102, 48, 11])
    result = profiler.score(entity, normal_test)
    print(f"Normal behavior: deviation={result['deviation_score']:.3f}, anomalous={result['is_anomalous']}")
    
    # Test anomalous behavior
    anomalous = np.array([500, 200, 50])  # Way outside normal
    result = profiler.score(entity, anomalous)
    print(f"Anomalous behavior: deviation={result['deviation_score']:.3f}, anomalous={result['is_anomalous']}")
