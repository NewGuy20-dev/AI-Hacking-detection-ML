"""Geolocation-based threat analysis and IP mapping."""
from collections import defaultdict


class GeoAnalyzer:
    """Analyze threats by geographic location."""
    
    # Sample IP ranges to country (simplified - real impl would use MaxMind/IP2Location)
    IP_GEO_SAMPLE = {
        '1.': 'CN', '14.': 'CN', '27.': 'CN', '36.': 'CN',
        '5.': 'RU', '31.': 'RU', '46.': 'RU', '77.': 'RU',
        '41.': 'NG', '105.': 'NG',
        '185.': 'EU', '193.': 'EU', '194.': 'EU',
        '8.': 'US', '12.': 'US', '15.': 'US', '20.': 'US',
        '192.168.': 'PRIVATE', '10.': 'PRIVATE', '172.': 'PRIVATE',
    }
    
    HIGH_RISK_COUNTRIES = {'CN', 'RU', 'KP', 'IR', 'NG', 'RO', 'UA', 'BR'}
    
    def __init__(self):
        self.country_stats = defaultdict(lambda: {'attacks': 0, 'total': 0, 'ips': set()})
        self.ip_cache = {}
    
    def get_country(self, ip: str) -> str:
        """Get country code for IP (simplified lookup)."""
        if ip in self.ip_cache:
            return self.ip_cache[ip]
        
        for prefix, country in self.IP_GEO_SAMPLE.items():
            if ip.startswith(prefix):
                self.ip_cache[ip] = country
                return country
        self.ip_cache[ip] = 'UNKNOWN'
        return 'UNKNOWN'
    
    def record(self, ip: str, is_attack: bool = False):
        """Record IP activity."""
        country = self.get_country(ip)
        self.country_stats[country]['total'] += 1
        self.country_stats[country]['ips'].add(ip)
        if is_attack:
            self.country_stats[country]['attacks'] += 1
    
    def get_risk_score(self, ip: str) -> dict:
        """Calculate geographic risk score for IP."""
        country = self.get_country(ip)
        
        base_risk = 0.7 if country in self.HIGH_RISK_COUNTRIES else 0.2
        if country == 'PRIVATE':
            base_risk = 0.1
        elif country == 'UNKNOWN':
            base_risk = 0.5
        
        # Adjust based on historical attacks from this country
        stats = self.country_stats[country]
        if stats['total'] > 0:
            attack_ratio = stats['attacks'] / stats['total']
            base_risk = 0.5 * base_risk + 0.5 * attack_ratio
        
        return {
            'ip': ip, 'country': country, 
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
                    'attack_ratio': round(stats['attacks']/stats['total'], 3) if stats['total'] else 0
                })
        return sorted(result, key=lambda x: x['attacks'], reverse=True)


if __name__ == '__main__':
    geo = GeoAnalyzer()
    
    # Simulate traffic
    geo.record('1.2.3.4', is_attack=True)   # CN
    geo.record('1.2.3.5', is_attack=True)   # CN
    geo.record('5.6.7.8', is_attack=True)   # RU
    geo.record('8.8.8.8', is_attack=False)  # US
    geo.record('192.168.1.1', is_attack=False)  # Private
    
    print("Risk Scores:")
    for ip in ['1.2.3.4', '8.8.8.8', '192.168.1.1']:
        print(f"  {ip}: {geo.get_risk_score(ip)}")
    
    print(f"\nThreat Map: {geo.get_threat_map()}")
