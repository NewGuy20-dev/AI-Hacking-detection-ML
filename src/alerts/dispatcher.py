"""Alert dispatcher with routing and rate limiting."""
import os
import yaml
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .channels import BaseChannel, ConsoleChannel, WebhookChannel, EmailChannel


class AlertDispatcher:
    """Routes alerts to configured channels based on severity."""
    
    def __init__(self, config_path: str = "config/alerts.yaml"):
        self.channels: Dict[str, BaseChannel] = {}
        self.routing: Dict[str, List[str]] = {}
        self._rate_limiter: Dict[str, List[float]] = defaultdict(list)
        self._alert_counter = 0
        self._rate_limit_max = 10
        self._rate_limit_window = 60  # seconds
        
        self._load_config(config_path)
    
    def _load_config(self, path: str):
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            # Default config
            self.channels['console'] = ConsoleChannel(enabled=True)
            self.routing = {
                'CRITICAL': ['console'], 'HIGH': ['console'],
                'MEDIUM': ['console'], 'LOW': ['console']
            }
            return
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Initialize channels
        channels_cfg = config.get('channels', {})
        
        if channels_cfg.get('console', {}).get('enabled', True):
            self.channels['console'] = ConsoleChannel(
                enabled=True,
                log_file=channels_cfg.get('console', {}).get('log_file')
            )
        
        webhook_cfg = channels_cfg.get('webhook', {})
        if webhook_cfg.get('enabled') and webhook_cfg.get('url'):
            url = os.environ.get('WEBHOOK_URL', webhook_cfg.get('url', ''))
            if url and not url.startswith('${'):
                self.channels['webhook'] = WebhookChannel(
                    url=url,
                    format=webhook_cfg.get('format', 'slack'),
                    timeout=webhook_cfg.get('timeout_seconds', 10),
                    retry_attempts=webhook_cfg.get('retry_attempts', 3)
                )
        
        email_cfg = channels_cfg.get('email', {})
        if email_cfg.get('enabled'):
            self.channels['email'] = EmailChannel(
                smtp_host=email_cfg.get('smtp_host', ''),
                smtp_port=email_cfg.get('smtp_port', 587),
                username=os.environ.get('SMTP_USER', email_cfg.get('username', '')),
                password=os.environ.get('SMTP_PASS', email_cfg.get('password', '')),
                from_address=email_cfg.get('from_address', ''),
                recipients=email_cfg.get('recipients', []),
                use_tls=email_cfg.get('use_tls', True)
            )
        
        # Load routing
        self.routing = config.get('routing', {
            'CRITICAL': ['webhook', 'email', 'console'],
            'HIGH': ['webhook', 'console'],
            'MEDIUM': ['console'],
            'LOW': ['console']
        })
        
        # Rate limiting
        rate_cfg = config.get('rate_limit', {})
        if rate_cfg.get('enabled', True):
            self._rate_limit_max = rate_cfg.get('max_per_minute', 10)
            self._rate_limit_window = rate_cfg.get('aggregation_window_seconds', 60)
    
    def send(self, alert: dict) -> dict:
        """Send alert to appropriate channels based on severity."""
        # Add ID and timestamp if missing
        if 'id' not in alert:
            alert['id'] = self._generate_alert_id()
        if 'timestamp' not in alert:
            alert['timestamp'] = datetime.now().isoformat()
        
        severity = alert.get('severity', 'MEDIUM')
        
        # Check rate limit
        if not self._check_rate_limit(severity):
            return {'success': False, 'reason': 'rate_limited', 'alert_id': alert['id']}
        
        # Get channels for this severity
        channel_names = self.routing.get(severity, ['console'])
        notified = []
        
        for name in channel_names:
            channel = self.channels.get(name)
            if channel and channel.is_enabled():
                if channel.send(alert):
                    notified.append(name)
        
        return {
            'success': len(notified) > 0,
            'alert_id': alert['id'],
            'channels_notified': notified
        }
    
    def send_from_prediction(self, prediction: dict, input_data: dict,
                             threshold: float = 0.8) -> Optional[dict]:
        """Create and send alert from prediction result."""
        confidence = prediction.get('confidence', [0])
        if hasattr(confidence, '__iter__'):
            confidence = float(confidence[0]) if len(confidence) > 0 else 0
        
        is_attack = prediction.get('is_attack', [0])
        if hasattr(is_attack, '__iter__'):
            is_attack = bool(is_attack[0]) if len(is_attack) > 0 else False
        
        if not is_attack or confidence < threshold:
            return None
        
        alert = {
            'severity': self._get_severity(confidence),
            'attack_type': self._detect_attack_type(input_data),
            'confidence': confidence,
            'source': {k: str(v)[:100] for k, v in input_data.items()}
        }
        
        return self.send(alert)
    
    def add_channel(self, name: str, channel: BaseChannel):
        """Register a new notification channel."""
        self.channels[name] = channel
    
    def _generate_alert_id(self) -> str:
        self._alert_counter += 1
        return f"ALT-{datetime.now().strftime('%Y%m%d')}-{self._alert_counter:04d}"
    
    def _check_rate_limit(self, severity: str) -> bool:
        now = datetime.now().timestamp()
        key = severity
        
        # Clean old entries
        self._rate_limiter[key] = [t for t in self._rate_limiter[key] 
                                    if now - t < self._rate_limit_window]
        
        if len(self._rate_limiter[key]) >= self._rate_limit_max:
            return False
        
        self._rate_limiter[key].append(now)
        return True
    
    def _get_severity(self, confidence: float) -> str:
        if confidence > 0.95: return 'CRITICAL'
        if confidence > 0.85: return 'HIGH'
        if confidence > 0.7: return 'MEDIUM'
        return 'LOW'
    
    def _detect_attack_type(self, input_data: dict) -> str:
        text = str(input_data.get('payloads', input_data.get('payload', ''))).lower()
        if any(p in text for p in ["'", "union", "select", "--"]): return "SQL_INJECTION"
        if any(p in text for p in ["<script", "onerror", "javascript:"]): return "XSS"
        if any(p in text for p in [";", "|", "`", "$("]): return "COMMAND_INJECTION"
        if 'urls' in input_data or 'url' in input_data: return "MALICIOUS_URL"
        return "UNKNOWN"
