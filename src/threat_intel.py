"""Threat intelligence integration for IOC lookups."""
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime


class ThreatIntelDB:
    """Local threat intelligence database."""
    
    def __init__(self, db_path: str = None):
        base = Path('/workspaces/AI-Hacking-detection-ML')
        self.db_path = db_path or str(base / 'data/threat_intel.db')
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''CREATE TABLE IF NOT EXISTS iocs (
            id INTEGER PRIMARY KEY,
            type TEXT NOT NULL,
            value TEXT NOT NULL UNIQUE,
            threat_type TEXT,
            confidence INTEGER DEFAULT 50,
            source TEXT,
            added_at TEXT
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_value ON iocs(value)')
        conn.commit()
        conn.close()
    
    def add_ioc(self, ioc_type: str, value: str, threat_type: str = None, 
                confidence: int = 50, source: str = 'manual'):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                'INSERT OR REPLACE INTO iocs (type, value, threat_type, confidence, source, added_at) VALUES (?, ?, ?, ?, ?, ?)',
                (ioc_type, value.lower(), threat_type, confidence, source, datetime.now().isoformat())
            )
            conn.commit()
        finally:
            conn.close()
    
    def lookup(self, value: str) -> dict:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute('SELECT type, threat_type, confidence, source FROM iocs WHERE value = ?', (value.lower(),))
        row = cur.fetchone()
        conn.close()
        
        if row:
            return {'found': True, 'type': row[0], 'threat_type': row[1], 'confidence': row[2], 'source': row[3]}
        return {'found': False}
    
    def bulk_add(self, iocs: list):
        """Add multiple IOCs: [{'type': 'ip', 'value': '1.2.3.4', ...}, ...]"""
        conn = sqlite3.connect(self.db_path)
        for ioc in iocs:
            try:
                conn.execute(
                    'INSERT OR IGNORE INTO iocs (type, value, threat_type, confidence, source, added_at) VALUES (?, ?, ?, ?, ?, ?)',
                    (ioc.get('type', 'unknown'), ioc['value'].lower(), ioc.get('threat_type'), 
                     ioc.get('confidence', 50), ioc.get('source', 'bulk'), datetime.now().isoformat())
                )
            except:
                pass
        conn.commit()
        conn.close()
    
    def load_sample_iocs(self):
        """Load sample malicious IOCs for testing."""
        sample_iocs = [
            {'type': 'ip', 'value': '185.220.101.1', 'threat_type': 'tor_exit', 'confidence': 90},
            {'type': 'ip', 'value': '45.33.32.156', 'threat_type': 'scanner', 'confidence': 80},
            {'type': 'domain', 'value': 'malware.tk', 'threat_type': 'malware', 'confidence': 95},
            {'type': 'domain', 'value': 'phishing-site.xyz', 'threat_type': 'phishing', 'confidence': 90},
            {'type': 'hash', 'value': 'd41d8cd98f00b204e9800998ecf8427e', 'threat_type': 'malware', 'confidence': 100},
        ]
        self.bulk_add(sample_iocs)
        print(f"Loaded {len(sample_iocs)} sample IOCs")
    
    def stats(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute('SELECT type, COUNT(*) FROM iocs GROUP BY type')
        stats = {row[0]: row[1] for row in cur.fetchall()}
        cur = conn.execute('SELECT COUNT(*) FROM iocs')
        stats['total'] = cur.fetchone()[0]
        conn.close()
        return stats


class ThreatIntel:
    """Threat intelligence lookup service."""
    
    def __init__(self):
        self.db = ThreatIntelDB()
    
    def check_ip(self, ip: str) -> dict:
        result = self.db.lookup(ip)
        result['indicator'] = ip
        result['indicator_type'] = 'ip'
        return result
    
    def check_domain(self, domain: str) -> dict:
        result = self.db.lookup(domain)
        result['indicator'] = domain
        result['indicator_type'] = 'domain'
        return result
    
    def check_hash(self, file_hash: str) -> dict:
        result = self.db.lookup(file_hash.lower())
        result['indicator'] = file_hash
        result['indicator_type'] = 'hash'
        return result
    
    def enrich(self, indicators: list) -> list:
        """Check multiple indicators."""
        results = []
        for ind in indicators:
            if self._is_ip(ind):
                results.append(self.check_ip(ind))
            elif self._is_hash(ind):
                results.append(self.check_hash(ind))
            else:
                results.append(self.check_domain(ind))
        return results
    
    def _is_ip(self, s: str) -> bool:
        import re
        return bool(re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', s))
    
    def _is_hash(self, s: str) -> bool:
        return len(s) in (32, 40, 64) and all(c in '0123456789abcdefABCDEF' for c in s)


if __name__ == '__main__':
    ti = ThreatIntel()
    ti.db.load_sample_iocs()
    
    print("\nThreat Intel Lookups:")
    test_indicators = ['185.220.101.1', 'google.com', 'malware.tk', 'd41d8cd98f00b204e9800998ecf8427e']
    for ind in test_indicators:
        result = ti.enrich([ind])[0]
        status = "🚨 MALICIOUS" if result['found'] else "✓ Clean"
        print(f"  {ind:40s} {status}")
    
    print(f"\nDB Stats: {ti.db.stats()}")
