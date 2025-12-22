"""Base class for notification channels."""
from abc import ABC, abstractmethod


class BaseChannel(ABC):
    """Abstract base class for notification channels."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    @abstractmethod
    def send(self, alert: dict) -> bool:
        """Send alert through this channel. Returns success status."""
        pass
    
    @abstractmethod
    def format_message(self, alert: dict) -> str:
        """Format alert for this specific channel."""
        pass
    
    def is_enabled(self) -> bool:
        return self.enabled
