"""Protocol fingerprinting for device and OS identification."""
from collections import defaultdict


class ProtocolFingerprinter:
    """Identify devices/OS from traffic patterns."""
    
    # Known fingerprint patterns (TTL, window size, options)
    FINGERPRINTS = {
        'linux': {'ttl_range': (64, 64), 'window_sizes': [5840, 14600, 29200]},
        'windows': {'ttl_range': (128, 128), 'window_sizes': [8192, 65535]},
        'macos': {'ttl_range': (64, 64), 'window_sizes': [65535]},
        'ios': {'ttl_range': (64, 64), 'window_sizes': [65535]},
        'android': {'ttl_range': (64, 64), 'window_sizes': [14600, 65535]},
        'router': {'ttl_range': (255, 255), 'window_sizes': [4128, 8192]},
        'iot': {'ttl_range': (64, 128), 'window_sizes': [1024, 2048, 4096]},
    }
    
    def __init__(self):
        self.device_profiles = defaultdict(lambda: {'os': None, 'confidence': 0, 'observations': []})
    
    def fingerprint(self, ip: str, ttl: int, window_size: int, 
                    mss: int = None, options: list = None) -> dict:
        """Identify OS/device from packet characteristics."""
        scores = {}
        
        for os_name, fp in self.FINGERPRINTS.items():
            score = 0
            # TTL match
            if fp['ttl_range'][0] <= ttl <= fp['ttl_range'][1]:
                score += 0.5
            # Window size match
            if window_size in fp['window_sizes']:
                score += 0.4
            elif any(abs(window_size - ws) < 1000 for ws in fp['window_sizes']):
                score += 0.2
            scores[os_name] = score
        
        best_match = max(scores, key=scores.get)
        confidence = scores[best_match]
        
        # Update profile
        profile = self.device_profiles[ip]
        profile['observations'].append({'ttl': ttl, 'window': window_size})
        if confidence > profile['confidence']:
            profile['os'] = best_match
            profile['confidence'] = confidence
        
        return {
            'ip': ip, 'os': best_match, 'confidence': round(confidence, 2),
            'ttl': ttl, 'window_size': window_size
        }
    
    def get_profile(self, ip: str) -> dict:
        """Get device profile for IP."""
        p = self.device_profiles[ip]
        return {'ip': ip, 'os': p['os'], 'confidence': p['confidence'], 
                'observations': len(p['observations'])}
    
    def detect_spoofing(self, ip: str, ttl: int, window_size: int) -> dict:
        """Detect potential IP spoofing via fingerprint mismatch."""
        profile = self.device_profiles[ip]
        if not profile['os'] or len(profile['observations']) < 3:
            return {'spoofing_detected': False, 'reason': 'insufficient_data'}
        
        # Check if current packet matches established profile
        current = self.fingerprint(ip, ttl, window_size)
        if current['os'] != profile['os'] and current['confidence'] > 0.5:
            return {
                'spoofing_detected': True,
                'expected_os': profile['os'],
                'current_os': current['os'],
                'reason': 'fingerprint_mismatch'
            }
        return {'spoofing_detected': False}


if __name__ == '__main__':
    fp = ProtocolFingerprinter()
    
    # Fingerprint some hosts
    print("Fingerprinting:")
    print(f"  Linux server: {fp.fingerprint('10.0.0.1', ttl=64, window_size=14600)}")
    print(f"  Windows PC: {fp.fingerprint('10.0.0.2', ttl=128, window_size=65535)}")
    print(f"  IoT device: {fp.fingerprint('10.0.0.3', ttl=64, window_size=2048)}")
    
    # Build profile then detect spoofing
    for _ in range(5):
        fp.fingerprint('10.0.0.1', ttl=64, window_size=14600)
    
    print(f"\nSpoofing check (normal): {fp.detect_spoofing('10.0.0.1', 64, 14600)}")
    print(f"Spoofing check (suspicious): {fp.detect_spoofing('10.0.0.1', 128, 65535)}")
