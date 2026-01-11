"""Human-readable indicators for explainability."""
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Indicator:
    """Single detection indicator."""
    name: str
    description: str
    severity: str  # low, medium, high, critical
    evidence: str
    confidence: float = 0.0


@dataclass 
class IndicatorSet:
    """Collection of indicators for a detection."""
    primary: List[Indicator] = field(default_factory=list)
    secondary: List[Indicator] = field(default_factory=list)
    fp_indicators: List[Indicator] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "primary": [vars(i) for i in self.primary],
            "secondary": [vars(i) for i in self.secondary],
            "fp_indicators": [vars(i) for i in self.fp_indicators]
        }


class PayloadIndicatorExtractor:
    """Extract indicators from payload analysis."""
    
    SQL_KEYWORDS = ['SELECT', 'UNION', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'FROM', 'WHERE', 'OR', 'AND']
    XSS_PATTERNS = ['<script', 'javascript:', 'onerror=', 'onload=', 'onclick=', '<img', '<iframe']
    CMD_PATTERNS = ['|', ';', '`', '$(',  '&&', '||', '../', 'cat ', 'ls ', 'wget ', 'curl ']
    
    def extract(self, text: str, score: float) -> IndicatorSet:
        indicators = IndicatorSet()
        text_upper = text.upper()
        
        # SQL Injection indicators
        sql_found = [kw for kw in self.SQL_KEYWORDS if kw in text_upper]
        if sql_found:
            indicators.primary.append(Indicator(
                name="SQL Keywords",
                description=f"SQL keywords detected: {', '.join(sql_found[:5])}",
                severity="high" if len(sql_found) >= 3 else "medium",
                evidence=text[:100],
                confidence=min(0.3 * len(sql_found), 0.9)
            ))
        
        # Check for SQL comment sequences
        if '--' in text or '/*' in text:
            indicators.primary.append(Indicator(
                name="SQL Comment",
                description="SQL comment sequence detected",
                severity="high",
                evidence=text[:100],
                confidence=0.7
            ))
        
        # XSS indicators
        xss_found = [p for p in self.XSS_PATTERNS if p.lower() in text.lower()]
        if xss_found:
            indicators.primary.append(Indicator(
                name="XSS Pattern",
                description=f"XSS patterns detected: {', '.join(xss_found[:3])}",
                severity="high",
                evidence=text[:100],
                confidence=0.8
            ))
        
        # Command injection indicators
        cmd_found = [p for p in self.CMD_PATTERNS if p in text]
        if cmd_found:
            indicators.primary.append(Indicator(
                name="Command Injection",
                description=f"Command injection patterns: {', '.join(cmd_found[:3])}",
                severity="critical" if '|' in text or ';' in text else "high",
                evidence=text[:100],
                confidence=0.75
            ))
        
        # Encoding anomalies
        if '%' in text and re.search(r'%[0-9a-fA-F]{2}', text):
            indicators.secondary.append(Indicator(
                name="URL Encoding",
                description="URL-encoded characters detected",
                severity="medium",
                evidence=text[:100],
                confidence=0.5
            ))
        
        # FP indicators
        if score < 0.7:
            if re.match(r'^[a-zA-Z0-9\s.,!?@#$%&*()-]+$', text):
                indicators.fp_indicators.append(Indicator(
                    name="Normal Characters",
                    description="Text contains only common characters",
                    severity="low",
                    evidence="",
                    confidence=0.6
                ))
        
        return indicators


class URLIndicatorExtractor:
    """Extract indicators from URL analysis."""
    
    SUSPICIOUS_TLDS = ['.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.top', '.work', '.click']
    
    def extract(self, url: str, score: float) -> IndicatorSet:
        indicators = IndicatorSet()
        
        # Suspicious TLD
        for tld in self.SUSPICIOUS_TLDS:
            if url.lower().endswith(tld):
                indicators.primary.append(Indicator(
                    name="Suspicious TLD",
                    description=f"URL uses suspicious TLD: {tld}",
                    severity="medium",
                    evidence=url,
                    confidence=0.6
                ))
                break
        
        # Excessive path depth
        path_depth = url.count('/')
        if path_depth > 6:
            indicators.secondary.append(Indicator(
                name="Deep Path",
                description=f"URL has excessive path depth: {path_depth} levels",
                severity="low",
                evidence=url,
                confidence=0.4
            ))
        
        # IP address in URL
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
            indicators.primary.append(Indicator(
                name="IP Address URL",
                description="URL contains IP address instead of domain",
                severity="high",
                evidence=url,
                confidence=0.7
            ))
        
        # Suspicious keywords
        suspicious = ['login', 'verify', 'secure', 'account', 'update', 'confirm']
        found = [s for s in suspicious if s in url.lower()]
        if found:
            indicators.secondary.append(Indicator(
                name="Phishing Keywords",
                description=f"Suspicious keywords in URL: {', '.join(found)}",
                severity="medium",
                evidence=url,
                confidence=0.5
            ))
        
        # Long URL
        if len(url) > 200:
            indicators.secondary.append(Indicator(
                name="Long URL",
                description=f"Unusually long URL: {len(url)} characters",
                severity="low",
                evidence=url[:100] + "...",
                confidence=0.3
            ))
        
        return indicators


class NetworkIndicatorExtractor:
    """Extract indicators from network flow analysis."""
    
    SUSPICIOUS_PORTS = [4444, 5555, 6666, 31337, 12345, 54321]
    
    def extract(self, flow_data: Dict, score: float) -> IndicatorSet:
        indicators = IndicatorSet()
        
        # Suspicious port
        dst_port = flow_data.get('dst_port', 0)
        if dst_port in self.SUSPICIOUS_PORTS:
            indicators.primary.append(Indicator(
                name="Suspicious Port",
                description=f"Connection to known malicious port: {dst_port}",
                severity="high",
                evidence=f"dst_port={dst_port}",
                confidence=0.8
            ))
        
        # High byte count
        bytes_sent = flow_data.get('bytes_sent', 0)
        if bytes_sent > 1000000:
            indicators.secondary.append(Indicator(
                name="High Data Transfer",
                description=f"Large data transfer: {bytes_sent/1024/1024:.1f} MB",
                severity="medium",
                evidence=f"bytes={bytes_sent}",
                confidence=0.5
            ))
        
        # Unusual protocol
        protocol = flow_data.get('protocol', '')
        if protocol.lower() in ['icmp', 'gre']:
            indicators.secondary.append(Indicator(
                name="Unusual Protocol",
                description=f"Uncommon protocol detected: {protocol}",
                severity="low",
                evidence=f"protocol={protocol}",
                confidence=0.4
            ))
        
        return indicators


def get_indicator_extractor(model_type: str):
    """Get appropriate indicator extractor for model type."""
    extractors = {
        'payload': PayloadIndicatorExtractor(),
        'content': PayloadIndicatorExtractor(),
        'url': URLIndicatorExtractor(),
        'network': NetworkIndicatorExtractor(),
    }
    return extractors.get(model_type, PayloadIndicatorExtractor())
