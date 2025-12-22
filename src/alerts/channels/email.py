"""Email notification channel."""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from .base import BaseChannel


class EmailChannel(BaseChannel):
    """Send alerts via SMTP email."""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str,
                 from_address: str, recipients: list, enabled: bool = True, use_tls: bool = True):
        super().__init__(enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self.recipients = recipients
        self.use_tls = use_tls
    
    def send(self, alert: dict) -> bool:
        if not self.enabled or not self.recipients:
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 {alert.get('severity', 'ALERT')}: {alert.get('attack_type', 'Security Alert')}"
            msg['From'] = self.from_address
            msg['To'] = ', '.join(self.recipients)
            
            msg.attach(MIMEText(self.format_message(alert), 'html'))
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.sendmail(self.from_address, self.recipients, msg.as_string())
            return True
        except Exception:
            return False
    
    def format_message(self, alert: dict) -> str:
        severity = alert.get('severity', 'UNKNOWN')
        colors = {'CRITICAL': '#FF0000', 'HIGH': '#FF6600', 'MEDIUM': '#FFCC00', 'LOW': '#00CC00'}
        
        return f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="border-left: 4px solid {colors.get(severity, '#808080')}; padding-left: 15px;">
                <h2 style="color: {colors.get(severity, '#808080')};">
                    🚨 {severity}: {alert.get('attack_type', 'Security Alert')}
                </h2>
                <p><strong>Confidence:</strong> {alert.get('confidence', 0):.1%}</p>
                <p><strong>Alert ID:</strong> {alert.get('id', 'N/A')}</p>
                <p><strong>Time:</strong> {alert.get('timestamp', 'N/A')}</p>
            </div>
        </body>
        </html>
        """
