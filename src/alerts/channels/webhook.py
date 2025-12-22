"""Webhook notification channel for Slack/Discord/Teams."""
import json
import requests
from .base import BaseChannel


class WebhookChannel(BaseChannel):
    """Send alerts via webhook (Slack, Discord, Teams)."""
    
    def __init__(self, url: str, format: str = "slack", enabled: bool = True,
                 timeout: int = 10, retry_attempts: int = 3):
        super().__init__(enabled)
        self.url = url
        self.format = format
        self.timeout = timeout
        self.retry_attempts = retry_attempts
    
    def send(self, alert: dict) -> bool:
        if not self.enabled or not self.url:
            return False
        
        payload = self._format_payload(alert)
        
        for attempt in range(self.retry_attempts):
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                if resp.status_code in (200, 204):
                    return True
            except Exception:
                continue
        return False
    
    def format_message(self, alert: dict) -> str:
        return json.dumps(self._format_payload(alert))
    
    def _format_payload(self, alert: dict) -> dict:
        if self.format == "discord":
            return self._format_discord(alert)
        return self._format_slack(alert)
    
    def _format_slack(self, alert: dict) -> dict:
        severity = alert.get('severity', 'UNKNOWN')
        colors = {'CRITICAL': '#FF0000', 'HIGH': '#FF6600', 'MEDIUM': '#FFCC00', 'LOW': '#00CC00'}
        
        return {
            "attachments": [{
                "color": colors.get(severity, '#808080'),
                "title": f"🚨 {severity}: {alert.get('attack_type', 'Alert')}",
                "fields": [
                    {"title": "Confidence", "value": f"{alert.get('confidence', 0):.1%}", "short": True},
                    {"title": "ID", "value": alert.get('id', 'N/A'), "short": True},
                ],
                "ts": alert.get('timestamp', '')
            }]
        }
    
    def _format_discord(self, alert: dict) -> dict:
        severity = alert.get('severity', 'UNKNOWN')
        colors = {'CRITICAL': 0xFF0000, 'HIGH': 0xFF6600, 'MEDIUM': 0xFFCC00, 'LOW': 0x00CC00}
        
        return {
            "embeds": [{
                "title": f"🚨 {severity}: {alert.get('attack_type', 'Alert')}",
                "color": colors.get(severity, 0x808080),
                "fields": [
                    {"name": "Confidence", "value": f"{alert.get('confidence', 0):.1%}", "inline": True},
                    {"name": "ID", "value": alert.get('id', 'N/A'), "inline": True},
                ]
            }]
        }
