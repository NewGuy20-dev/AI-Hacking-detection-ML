"""File hash reputation checker."""
import hashlib
from pathlib import Path
from threat_intel import ThreatIntel


class HashChecker:
    """Check file hashes against known malware databases."""
    
    def __init__(self):
        self.ti = ThreatIntel()
    
    def hash_file(self, filepath: str) -> dict:
        """Calculate hashes for a file."""
        path = Path(filepath)
        if not path.exists():
            return {'error': 'File not found'}
        
        md5 = hashlib.md5()
        sha1 = hashlib.sha1()
        sha256 = hashlib.sha256()
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
                sha1.update(chunk)
                sha256.update(chunk)
        
        return {
            'filename': path.name,
            'size': path.stat().st_size,
            'md5': md5.hexdigest(),
            'sha1': sha1.hexdigest(),
            'sha256': sha256.hexdigest()
        }
    
    def check_hash(self, file_hash: str) -> dict:
        """Check a hash against threat intel."""
        result = self.ti.check_hash(file_hash)
        result['hash'] = file_hash
        return result
    
    def check_file(self, filepath: str) -> dict:
        """Hash a file and check all hashes."""
        hashes = self.hash_file(filepath)
        if 'error' in hashes:
            return hashes
        
        # Check each hash
        for hash_type in ['md5', 'sha1', 'sha256']:
            result = self.check_hash(hashes[hash_type])
            if result['found']:
                return {
                    **hashes,
                    'malicious': True,
                    'threat_type': result.get('threat_type'),
                    'confidence': result.get('confidence'),
                    'matched_hash': hash_type
                }
        
        return {**hashes, 'malicious': False}


if __name__ == '__main__':
    checker = HashChecker()
    
    # Test with known malicious hash
    print("Hash Reputation Check:")
    test_hashes = [
        'd41d8cd98f00b204e9800998ecf8427e',  # In our sample DB
        'e3b0c44298fc1c149afbf4c8996fb924',  # Clean
    ]
    
    for h in test_hashes:
        result = checker.check_hash(h)
        status = "🚨 MALICIOUS" if result['found'] else "✓ Clean"
        print(f"  {h} {status}")
