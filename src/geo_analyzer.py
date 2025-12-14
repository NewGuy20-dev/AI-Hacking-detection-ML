"""Geolocation-based threat analysis using IP2Location LITE database."""
import csv
import struct
import socket
from collections import defaultdict
from pathlib import Path


class IP2LocationDB:
    """IP2Location LITE database lookup."""
    
    def __init__(self, db_path: str = None):
        base = Path('/workspaces/AI-Hacking-detection-ML')
        self.db_path = db_path or str(base / 'data/IP2LOCATION-LITE-DB1.CSV')
        self.ranges = []  # [(start_int, end_int, country_code, country_name)]
        self._load_db()
    
    def _load_db(self):
        """Load IP ranges from CSV."""
        if not Path(self.db_path).exists():
            print(f"Warning: IP2Location DB not found at {self.db_path}")
            return
        
        with open(self.db_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4:
                    self.ranges.append((int(row[0]), int(row[1]), row[2], row[3]))
        
        # Sort by start IP for binary search
        self.ranges.sort(key=lambda x: x[0])
    
    def _ip_to_int(self, ip: str) -> int:
        """Convert IP string to integer."""
        try:
            return struct.unpack("!I", socket.inet_aton(ip))[0]
        except:
            return 0
    
    def lookup(self, ip: str) -> dict:
        """Lookup country for IP address."""
        # Handle private IPs
        if ip.startswith(('10.', '192.168.', '172.16.', '172.17.', '172.18.', 
                          '172.19.', '172.2', '172.30.', '172.31.', '127.')):
            return {'ip': ip, 'country_code': 'PRIVATE', 'country_name': 'Private Network'}
        
        ip_int = self._ip_to_int(ip)
        if ip_int == 0:
            return {'ip': ip, 'country_code': 'INVALID', 'country_name': 'Invalid IP'}
        
        # Binary search
        left, right = 0, len(self.ranges) - 1
        while left <= right:
            mid = (left + right) // 2
            start, end, code, name = self.ranges[mid]
            if start <= ip_int <= end:
                return {'ip': ip, 'country_code': code, 'country_name': name}
            elif ip_int < start:
                right = mid - 1
            else:
                left = mid + 1
        
        return {'ip': ip, 'country_code': 'UNKNOWN', 'country_name': 'Unknown'}


class GeoAnalyzer:
    """Analyze threats by geographic location."""
    
    HIGH_RISK_COUNTRIES = {'CN', 'RU', 'KP', 'IR', 'NG', 'RO', 'UA', 'BR', 'VN', 'IN'}
    
    def __init__(self):
        self.db = IP2LocationDB()
        self.country_stats = defaultdict(lambda: {'attacks': 0, 'total': 0, 'ips': set()})
    
    def get_country(self, ip: str) -> dict:
        """Get country info for IP."""
        return self.db.lookup(ip)
    
    def record(self, ip: str, is_attack: bool = False):
        """Record IP activity."""
        info = self.get_country(ip)
        country = info['country_code']
        self.country_stats[country]['total'] += 1
        self.country_stats[country]['ips'].add(ip)
        if is_attack:
            self.country_stats[country]['attacks'] += 1
    
    def get_risk_score(self, ip: str) -> dict:
        """Calculate geographic risk score for IP."""
        info = self.get_country(ip)
        country = info['country_code']
        
        base_risk = 0.7 if country in self.HIGH_RISK_COUNTRIES else 0.2
        if country == 'PRIVATE':
            base_risk = 0.1
        elif country in ('UNKNOWN', 'INVALID'):
            base_risk = 0.5
        
        # Adjust based on historical attacks
        stats = self.country_stats[country]
        if stats['total'] > 0:
            attack_ratio = stats['attacks'] / stats['total']
            base_risk = 0.5 * base_risk + 0.5 * attack_ratio
        
        return {
            'ip': ip,
            'country_code': country,
            'country_name': info['country_name'],
            'risk_score': round(base_risk, 3),
            'high_risk_country': country in self.HIGH_RISK_COUNTRIES
        }
    
    def get_threat_map(self) -> list:
        """Get attack statistics by country."""
        result = []
        for country, stats in self.country_stats.items():
            if stats['attacks'] > 0:
                result.append({
                    'country': country,
                    'attacks': stats['attacks'],
                    'unique_ips': len(stats['ips']),
                    'attack_ratio': round(stats['attacks']/stats['total'], 3)
                })
        return sorted(result, key=lambda x: x['attacks'], reverse=True)


if __name__ == '__main__':
    geo = GeoAnalyzer()
    
    # Test lookups
    test_ips = [
        '8.8.8.8',        # Google DNS (US)
        '1.1.1.1',        # Cloudflare (AU)
        '114.114.114.114', # China
        '77.88.8.8',      # Yandex (RU)
        '192.168.1.1',    # Private
    ]
    
    print("IP2Location Lookups:")
    for ip in test_ips:
        result = geo.get_risk_score(ip)
        risk = "⚠️ HIGH RISK" if result['high_risk_country'] else "✓"
        print(f"  {ip:18s} -> {result['country_code']:3s} ({result['country_name'][:20]:20s}) risk={result['risk_score']:.2f} {risk}")
