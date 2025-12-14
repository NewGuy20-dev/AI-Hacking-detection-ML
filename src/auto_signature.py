"""Auto-signature generation for dynamic detection rules."""
import numpy as np
from collections import defaultdict
from datetime import datetime


class AutoSignatureGenerator:
    """Generate detection signatures from observed attacks."""
    
    def __init__(self, min_samples: int = 5):
        self.attack_samples = defaultdict(list)
        self.signatures = {}
        self.min_samples = min_samples
    
    def record_attack(self, attack_type: str, features: dict):
        """Record attack sample for signature generation."""
        self.attack_samples[attack_type].append(features)
    
    def generate_signature(self, attack_type: str) -> dict:
        """Generate signature from attack samples."""
        samples = self.attack_samples[attack_type]
        if len(samples) < self.min_samples:
            return None
        
        # Find common patterns
        signature = {
            'attack_type': attack_type,
            'created': datetime.now().isoformat(),
            'sample_count': len(samples),
            'rules': []
        }
        
        # Analyze each feature
        all_keys = set()
        for s in samples:
            all_keys.update(s.keys())
        
        for key in all_keys:
            values = [s.get(key) for s in samples if key in s]
            if not values:
                continue
            
            # Numeric features: find range
            if all(isinstance(v, (int, float)) for v in values):
                min_val, max_val = min(values), max(values)
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Create range rule with some tolerance
                signature['rules'].append({
                    'field': key,
                    'type': 'range',
                    'min': round(mean_val - 2*std_val, 2),
                    'max': round(mean_val + 2*std_val, 2),
                    'typical': round(mean_val, 2)
                })
            
            # Categorical: find common values
            else:
                value_counts = defaultdict(int)
                for v in values:
                    value_counts[v] += 1
                
                common = [v for v, c in value_counts.items() if c >= len(samples) * 0.5]
                if common:
                    signature['rules'].append({
                        'field': key,
                        'type': 'match',
                        'values': common
                    })
        
        self.signatures[attack_type] = signature
        return signature
    
    def match(self, features: dict) -> list:
        """Check if features match any signature."""
        matches = []
        
        for attack_type, sig in self.signatures.items():
            score = 0
            total_rules = len(sig['rules'])
            
            for rule in sig['rules']:
                field = rule['field']
                if field not in features:
                    continue
                
                value = features[field]
                
                if rule['type'] == 'range':
                    if rule['min'] <= value <= rule['max']:
                        score += 1
                elif rule['type'] == 'match':
                    if value in rule['values']:
                        score += 1
            
            if total_rules > 0 and score / total_rules >= 0.6:
                matches.append({
                    'attack_type': attack_type,
                    'confidence': round(score / total_rules, 2),
                    'rules_matched': score,
                    'total_rules': total_rules
                })
        
        return sorted(matches, key=lambda x: x['confidence'], reverse=True)
    
    def export_rules(self) -> list:
        """Export signatures as detection rules."""
        rules = []
        for attack_type, sig in self.signatures.items():
            rule = {
                'name': f'AUTO_{attack_type.upper()}',
                'description': f'Auto-generated rule for {attack_type}',
                'conditions': [],
                'action': 'alert',
                'severity': 'medium'
            }
            
            for r in sig['rules']:
                if r['type'] == 'range':
                    rule['conditions'].append(f"{r['field']} >= {r['min']} AND {r['field']} <= {r['max']}")
                else:
                    rule['conditions'].append(f"{r['field']} IN {r['values']}")
            
            rules.append(rule)
        return rules


if __name__ == '__main__':
    gen = AutoSignatureGenerator(min_samples=3)
    
    # Record DoS attack samples
    for i in range(5):
        gen.record_attack('DoS', {
            'packets_per_sec': 50000 + np.random.randint(-5000, 5000),
            'src_port': 'random',
            'protocol': 'TCP',
            'payload_size': 1400 + np.random.randint(-100, 100)
        })
    
    # Record Probe samples
    for i in range(5):
        gen.record_attack('Probe', {
            'packets_per_sec': 100 + np.random.randint(-20, 20),
            'unique_ports': 1000 + np.random.randint(-200, 200),
            'protocol': 'TCP'
        })
    
    # Generate signatures
    print("Generated Signatures:")
    for attack_type in ['DoS', 'Probe']:
        sig = gen.generate_signature(attack_type)
        print(f"\n{attack_type}: {len(sig['rules'])} rules")
        for r in sig['rules']:
            print(f"  - {r}")
    
    # Test matching
    print("\nMatching test:")
    test = {'packets_per_sec': 48000, 'protocol': 'TCP', 'payload_size': 1350}
    matches = gen.match(test)
    print(f"  Input: {test}")
    print(f"  Matches: {matches}")
