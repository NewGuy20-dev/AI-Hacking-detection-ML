"""Console notification channel."""
import logging
from datetime import datetime
from .base import BaseChannel


class ConsoleChannel(BaseChannel):
    """Print alerts to console/log."""
    
    def __init__(self, enabled: bool = True, log_file: str = None):
        super().__init__(enabled)
        self.logger = logging.getLogger('alerts')
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def send(self, alert: dict) -> bool:
        if not self.enabled:
            return False
        msg = self.format_message(alert)
        print(msg)
        self.logger.info(msg)
        return True
    
    def format_message(self, alert: dict) -> str:
        severity = alert.get('severity') or 'UNKNOWN'
        icons = {'CRITICAL': '🚨', 'HIGH': '⚠️', 'MEDIUM': '⚡', 'LOW': 'ℹ️'}
        icon = icons.get(severity, '•')
        
        confidence = alert.get('confidence') or 0
        
        return (
            f"{icon} [{severity}] {alert.get('attack_type') or 'ALERT'}\n"
            f"   Confidence: {confidence:.1%}\n"
            f"   Time: {alert.get('timestamp', datetime.now().isoformat())}\n"
            f"   ID: {alert.get('id', 'N/A')}"
        )
