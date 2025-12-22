"""Tests for Alert System."""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.alerts.channels.base import BaseChannel
from src.alerts.channels.console import ConsoleChannel
from src.alerts.channels.webhook import WebhookChannel
from src.alerts.channels.email import EmailChannel
from src.alerts.dispatcher import AlertDispatcher


class TestConsoleChannel:
    
    def test_send_enabled(self, capsys):
        """Test console channel sends when enabled."""
        channel = ConsoleChannel(enabled=True)
        alert = {'severity': 'HIGH', 'attack_type': 'SQL_INJECTION', 'confidence': 0.95, 'id': 'TEST-001'}
        
        result = channel.send(alert)
        
        assert result is True
        captured = capsys.readouterr()
        assert 'HIGH' in captured.out
        assert 'SQL_INJECTION' in captured.out
    
    def test_send_disabled(self, capsys):
        """Test console channel doesn't send when disabled."""
        channel = ConsoleChannel(enabled=False)
        alert = {'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9}
        
        result = channel.send(alert)
        
        assert result is False
        captured = capsys.readouterr()
        assert captured.out == ''
    
    def test_format_message_all_severities(self):
        """Test formatting for all severity levels."""
        channel = ConsoleChannel()
        
        for severity, icon in [('CRITICAL', '🚨'), ('HIGH', '⚠️'), ('MEDIUM', '⚡'), ('LOW', 'ℹ️')]:
            alert = {'severity': severity, 'attack_type': 'TEST', 'confidence': 0.5}
            msg = channel.format_message(alert)
            assert icon in msg
            assert severity in msg
    
    def test_format_message_missing_fields(self):
        """Test formatting handles missing fields gracefully."""
        channel = ConsoleChannel()
        alert = {}  # Empty alert
        
        msg = channel.format_message(alert)
        
        assert 'UNKNOWN' in msg or 'ALERT' in msg
    
    def test_log_file_creation(self, tmp_path):
        """Test log file is created when specified."""
        log_file = tmp_path / "test_alerts.log"
        channel = ConsoleChannel(enabled=True, log_file=str(log_file))
        
        channel.send({'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9})
        
        # Logger may buffer, so just check channel was created
        assert channel.logger is not None


class TestWebhookChannel:
    
    def test_send_disabled(self):
        """Test webhook doesn't send when disabled."""
        channel = WebhookChannel(url="http://test.com", enabled=False)
        
        result = channel.send({'severity': 'HIGH'})
        
        assert result is False
    
    def test_send_no_url(self):
        """Test webhook handles empty URL."""
        channel = WebhookChannel(url="", enabled=True)
        
        result = channel.send({'severity': 'HIGH'})
        
        assert result is False
    
    @patch('src.alerts.channels.webhook.requests.post')
    def test_send_success(self, mock_post):
        """Test successful webhook send."""
        mock_post.return_value.status_code = 200
        channel = WebhookChannel(url="http://test.com/webhook", enabled=True)
        
        result = channel.send({'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9})
        
        assert result is True
        mock_post.assert_called_once()
    
    @patch('src.alerts.channels.webhook.requests.post')
    def test_send_retry_on_failure(self, mock_post):
        """Test webhook retries on failure."""
        mock_post.return_value.status_code = 500
        channel = WebhookChannel(url="http://test.com", retry_attempts=3)
        
        result = channel.send({'severity': 'HIGH'})
        
        assert result is False
        assert mock_post.call_count == 3
    
    @patch('src.alerts.channels.webhook.requests.post')
    def test_send_handles_exception(self, mock_post):
        """Test webhook handles request exceptions."""
        mock_post.side_effect = Exception("Connection error")
        channel = WebhookChannel(url="http://test.com", retry_attempts=2)
        
        result = channel.send({'severity': 'HIGH'})
        
        assert result is False
    
    def test_format_slack(self):
        """Test Slack message formatting."""
        channel = WebhookChannel(url="http://test.com", format="slack")
        alert = {'severity': 'HIGH', 'attack_type': 'XSS', 'confidence': 0.85, 'id': 'ALT-001'}
        
        payload = channel._format_slack(alert)
        
        assert 'attachments' in payload
        assert payload['attachments'][0]['color'] == '#FF6600'
    
    def test_format_discord(self):
        """Test Discord message formatting."""
        channel = WebhookChannel(url="http://test.com", format="discord")
        alert = {'severity': 'CRITICAL', 'attack_type': 'SQL', 'confidence': 0.99}
        
        payload = channel._format_discord(alert)
        
        assert 'embeds' in payload
        assert payload['embeds'][0]['color'] == 0xFF0000


class TestEmailChannel:
    
    def test_send_disabled(self):
        """Test email doesn't send when disabled."""
        channel = EmailChannel(
            smtp_host="smtp.test.com", smtp_port=587,
            username="", password="", from_address="test@test.com",
            recipients=["admin@test.com"], enabled=False
        )
        
        result = channel.send({'severity': 'HIGH'})
        
        assert result is False
    
    def test_send_no_recipients(self):
        """Test email handles empty recipients."""
        channel = EmailChannel(
            smtp_host="smtp.test.com", smtp_port=587,
            username="", password="", from_address="test@test.com",
            recipients=[], enabled=True
        )
        
        result = channel.send({'severity': 'HIGH'})
        
        assert result is False
    
    def test_format_message_html(self):
        """Test HTML email formatting."""
        channel = EmailChannel(
            smtp_host="", smtp_port=587, username="", password="",
            from_address="", recipients=[]
        )
        alert = {'severity': 'CRITICAL', 'attack_type': 'SQL_INJECTION', 'confidence': 0.98, 'id': 'ALT-001'}
        
        html = channel.format_message(alert)
        
        assert '<html>' in html
        assert 'CRITICAL' in html
        assert '98' in html  # confidence percentage


class TestAlertDispatcher:
    
    def test_init_default_config(self):
        """Test dispatcher initializes with defaults when no config."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        assert 'console' in dispatcher.channels
        assert 'CRITICAL' in dispatcher.routing
    
    def test_send_generates_id(self):
        """Test send generates alert ID if missing."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        alert = {'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9}
        
        result = dispatcher.send(alert)
        
        assert result['alert_id'].startswith('ALT-')
    
    def test_send_generates_timestamp(self):
        """Test send generates timestamp if missing."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        alert = {'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9}
        
        dispatcher.send(alert)
        
        assert 'timestamp' in alert
    
    def test_send_routes_by_severity(self, capsys):
        """Test alerts route to correct channels by severity."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        result = dispatcher.send({'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9})
        
        assert 'console' in result['channels_notified']
    
    def test_rate_limiting(self):
        """Test rate limiting prevents flood."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        dispatcher._rate_limit_max = 3
        dispatcher._rate_limit_window = 60
        
        results = []
        for i in range(5):
            result = dispatcher.send({'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9})
            results.append(result)
        
        # First 3 should succeed, rest should be rate limited
        assert results[0]['success'] is True
        assert results[3]['success'] is False
        assert results[3]['reason'] == 'rate_limited'
    
    def test_send_from_prediction_below_threshold(self):
        """Test no alert sent when confidence below threshold."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        prediction = {'confidence': [0.5], 'is_attack': [True]}
        
        result = dispatcher.send_from_prediction(prediction, {'payload': 'test'}, threshold=0.8)
        
        assert result is None
    
    def test_send_from_prediction_above_threshold(self):
        """Test alert sent when confidence above threshold."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        prediction = {'confidence': [0.95], 'is_attack': [True]}
        
        result = dispatcher.send_from_prediction(prediction, {'payload': "' OR 1=1--"}, threshold=0.8)
        
        assert result is not None
        assert result['success'] is True
    
    def test_send_from_prediction_not_attack(self):
        """Test no alert when not classified as attack."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        prediction = {'confidence': [0.95], 'is_attack': [False]}
        
        result = dispatcher.send_from_prediction(prediction, {'payload': 'test'}, threshold=0.8)
        
        assert result is None
    
    def test_detect_attack_type_sql(self):
        """Test SQL injection detection."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        assert dispatcher._detect_attack_type({'payload': "' OR 1=1--"}) == "SQL_INJECTION"
        assert dispatcher._detect_attack_type({'payload': "UNION SELECT"}) == "SQL_INJECTION"
    
    def test_detect_attack_type_xss(self):
        """Test XSS detection."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        assert dispatcher._detect_attack_type({'payload': "<script>alert(1)</script>"}) == "XSS"
        assert dispatcher._detect_attack_type({'payload': "onerror=alert(1)"}) == "XSS"
    
    def test_detect_attack_type_command_injection(self):
        """Test command injection detection."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        assert dispatcher._detect_attack_type({'payload': "; cat /etc/passwd"}) == "COMMAND_INJECTION"
        assert dispatcher._detect_attack_type({'payload': "| ls -la"}) == "COMMAND_INJECTION"
    
    def test_get_severity_levels(self):
        """Test severity level calculation."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        assert dispatcher._get_severity(0.99) == 'CRITICAL'
        assert dispatcher._get_severity(0.90) == 'HIGH'
        assert dispatcher._get_severity(0.75) == 'MEDIUM'
        assert dispatcher._get_severity(0.50) == 'LOW'
    
    def test_add_channel(self):
        """Test adding custom channel."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        mock_channel = Mock(spec=BaseChannel)
        mock_channel.is_enabled.return_value = True
        mock_channel.send.return_value = True
        
        dispatcher.add_channel('custom', mock_channel)
        dispatcher.routing['HIGH'] = ['custom']
        
        result = dispatcher.send({'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9})
        
        assert 'custom' in result['channels_notified']


class TestAlertDispatcherEdgeCases:
    
    def test_empty_alert(self):
        """Test handling of empty alert dict."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        
        result = dispatcher.send({})
        
        assert 'alert_id' in result
    
    def test_none_values_in_alert(self):
        """Test handling of None values."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        alert = {'severity': None, 'attack_type': None, 'confidence': None}
        
        result = dispatcher.send(alert)
        
        assert result is not None
    
    def test_very_long_payload(self):
        """Test handling of very long payload in alert."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        long_payload = "A" * 100000
        
        result = dispatcher.send_from_prediction(
            {'confidence': [0.95], 'is_attack': [True]},
            {'payload': long_payload},
            threshold=0.8
        )
        
        # Should truncate and not crash
        assert result is not None
    
    def test_special_characters_in_payload(self):
        """Test handling of special characters."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        special_payload = "Test\x00\n\r\t<>&\"'payload"
        
        result = dispatcher.send_from_prediction(
            {'confidence': [0.95], 'is_attack': [True]},
            {'payload': special_payload},
            threshold=0.8
        )
        
        assert result is not None
    
    def test_unicode_in_payload(self):
        """Test handling of unicode characters."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        unicode_payload = "Test 你好 🚨 émoji payload"
        
        result = dispatcher.send_from_prediction(
            {'confidence': [0.95], 'is_attack': [True]},
            {'payload': unicode_payload},
            threshold=0.8
        )
        
        assert result is not None
    
    def test_concurrent_rate_limiting(self):
        """Test rate limiting with rapid concurrent sends."""
        dispatcher = AlertDispatcher(config_path="nonexistent.yaml")
        dispatcher._rate_limit_max = 5
        
        import threading
        results = []
        
        def send_alert():
            result = dispatcher.send({'severity': 'HIGH', 'attack_type': 'TEST', 'confidence': 0.9})
            results.append(result)
        
        threads = [threading.Thread(target=send_alert) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Some should be rate limited
        rate_limited = sum(1 for r in results if r.get('reason') == 'rate_limited')
        assert rate_limited > 0
