"""Discord webhook alerting with full metrics reporting."""
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class AlertConfig:
    """Alert configuration."""
    webhook_url: str
    accuracy_threshold: float = 0.99
    fp_threshold: float = 0.03
    fn_threshold: float = 0.02
    latency_threshold_ms: float = 100.0
    calibration_threshold: float = 0.1
    warn_ratio: float = 0.8  # Warn when metric is at 80% of threshold


class DiscordAlerter:
    """Send alerts and reports to Discord."""
    
    # Default webhook - always used
    DEFAULT_WEBHOOK = "https://discord.com/api/webhooks/1452715933398466782/Ajftu5_fHelFqifTRcZN3S7fCDddXPs89p9w8dTHX8pF1xUO59ckac_DyCTQsRKC1H8O"
    
    COLORS = {
        'success': 0x2ecc71,  # Green
        'warning': 0xf39c12,  # Orange
        'error': 0xe74c3c,    # Red
        'info': 0x3498db,     # Blue
    }
    
    def __init__(self, webhook_url: str = None, config: AlertConfig = None):
        self.webhook_url = webhook_url or os.environ.get('DISCORD_WEBHOOK_URL') or self.DEFAULT_WEBHOOK
        self.config = config or AlertConfig(webhook_url=self.webhook_url)
    
    def _send(self, payload: dict) -> bool:
        """Send payload to Discord webhook."""
        if not self.webhook_url or not HAS_REQUESTS:
            print("Discord webhook not configured or requests not available")
            return False
        
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"Discord send error: {e}")
            return False
    
    def _format_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as Discord-friendly table."""
        lines = ["```"]
        lines.append("┌────────────────┬─────────┬────────┬────────┐")
        lines.append("│ Metric         │ Value   │ Target │ Status │")
        lines.append("├────────────────┼─────────┼────────┼────────┤")
        
        # Accuracy
        acc = metrics.get('accuracy', 0) * 100
        acc_pass = acc >= self.config.accuracy_threshold * 100
        status = "✅" if acc_pass else "❌"
        lines.append(f"│ Accuracy       │ {acc:5.2f}%  │ >99%   │   {status}   │")
        
        # FP Rate
        fp = metrics.get('fp_rate', 0) * 100
        fp_pass = fp <= self.config.fp_threshold * 100
        status = "✅" if fp_pass else "❌"
        lines.append(f"│ FP Rate        │ {fp:5.2f}%  │ <3%    │   {status}   │")
        
        # FN Rate
        fn = metrics.get('fn_rate', 0) * 100
        fn_pass = fn <= self.config.fn_threshold * 100
        status = "✅" if fn_pass else "❌"
        lines.append(f"│ FN Rate        │ {fn:5.2f}%  │ <2%    │   {status}   │")
        
        # Latency (if present)
        if 'p95_latency_ms' in metrics:
            lat = metrics['p95_latency_ms']
            lat_pass = lat <= self.config.latency_threshold_ms
            status = "✅" if lat_pass else "❌"
            lines.append(f"│ P95 Latency    │ {lat:5.0f}ms │ <100ms │   {status}   │")
        
        # Calibration (if present)
        if 'calibration_error' in metrics:
            cal = metrics['calibration_error']
            cal_pass = cal <= self.config.calibration_threshold
            status = "✅" if cal_pass else "❌"
            lines.append(f"│ Calibration    │ {cal:5.3f}   │ <0.1   │   {status}   │")
        
        lines.append("└────────────────┴─────────┴────────┴────────┘")
        lines.append("```")
        
        return "\n".join(lines)
    
    def _format_category_breakdown(self, categories: Dict[str, Dict]) -> str:
        """Format per-category breakdown."""
        if not categories:
            return ""
        
        lines = ["**Category Breakdown:**", "```"]
        for cat, cat_metrics in sorted(categories.items()):
            fp = cat_metrics.get('fp_rate', 0) * 100
            acc = cat_metrics.get('accuracy', 0) * 100
            warn = "⚠️" if fp > 2.5 else ""
            lines.append(f"{cat:12s}: {acc:5.1f}% acc, {fp:4.1f}% FP {warn}")
        lines.append("```")
        
        return "\n".join(lines)
    
    def _format_trend(self, trend: List[float], metric_name: str = "FP Rate") -> str:
        """Format 7-day trend."""
        if not trend or len(trend) < 2:
            return ""
        
        trend_str = " → ".join([f"{v*100:.1f}%" for v in trend[-7:]])
        direction = "↓" if trend[-1] < trend[0] else "↑" if trend[-1] > trend[0] else "→"
        status = "improving" if direction == "↓" else "worsening" if direction == "↑" else "stable"
        
        return f"**7-Day Trend ({metric_name}):** {trend_str} ({status} {direction})"
    
    def send_success(self, 
                     metrics: Dict[str, Any],
                     trend: List[float] = None,
                     coverage: Dict[str, int] = None,
                     model_name: str = "payload_cnn") -> bool:
        """Send success report with full metrics."""
        embed = {
            "title": "✅ DAILY STRESS TEST PASSED",
            "color": self.COLORS['success'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [],
        }
        
        # Date and model
        embed["description"] = f"**Date:** {datetime.now().strftime('%Y-%m-%d')} | **Model:** {model_name}"
        
        # Metrics table
        embed["fields"].append({
            "name": "📊 METRICS",
            "value": self._format_metrics_table(metrics),
            "inline": False,
        })
        
        # Trend
        if trend:
            embed["fields"].append({
                "name": "📈 TREND",
                "value": self._format_trend(trend),
                "inline": False,
            })
        
        # Coverage
        if coverage:
            total = sum(coverage.values())
            coverage_str = f"• Samples tested: {total:,}\n"
            coverage_str += f"• Categories: {len(coverage)} | All passed ✅"
            embed["fields"].append({
                "name": "🧪 TEST COVERAGE",
                "value": coverage_str,
                "inline": False,
            })
        
        return self._send({"embeds": [embed]})
    
    def send_warning(self,
                     metrics: Dict[str, Any],
                     warnings: List[str],
                     categories: Dict[str, Dict] = None,
                     model_name: str = "payload_cnn") -> bool:
        """Send warning alert with metrics."""
        embed = {
            "title": "⚠️ STRESS TEST WARNING",
            "color": self.COLORS['warning'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [],
        }
        
        embed["description"] = f"**Date:** {datetime.now().strftime('%Y-%m-%d')} | **Severity:** WARNING"
        
        # Metrics table
        embed["fields"].append({
            "name": "📊 METRICS",
            "value": self._format_metrics_table(metrics),
            "inline": False,
        })
        
        # Warnings
        if warnings:
            warn_str = "\n".join([f"• {w}" for w in warnings])
            embed["fields"].append({
                "name": "⚠️ WARNINGS",
                "value": warn_str,
                "inline": False,
            })
        
        # Category breakdown
        if categories:
            embed["fields"].append({
                "name": "📉 CATEGORY ISSUES",
                "value": self._format_category_breakdown(categories),
                "inline": False,
            })
        
        # Recommendation
        embed["fields"].append({
            "name": "💡 RECOMMENDATION",
            "value": "Monitor affected categories. Consider adding samples to training.",
            "inline": False,
        })
        
        return self._send({"embeds": [embed]})
    
    def send_failure(self,
                     metrics: Dict[str, Any],
                     failures: List[str],
                     failed_samples: List[Dict] = None,
                     actions: List[str] = None,
                     model_name: str = "payload_cnn") -> bool:
        """Send failure alert with metrics and failed samples."""
        embed = {
            "title": "🚨 STRESS TEST FAILED",
            "color": self.COLORS['error'],
            "timestamp": datetime.utcnow().isoformat(),
            "fields": [],
        }
        
        embed["description"] = f"**Date:** {datetime.now().strftime('%Y-%m-%d')} | **Severity:** ERROR"
        
        # Metrics table
        embed["fields"].append({
            "name": "📊 METRICS",
            "value": self._format_metrics_table(metrics),
            "inline": False,
        })
        
        # Failures
        if failures:
            fail_str = "\n".join([f"• {f}" for f in failures])
            embed["fields"].append({
                "name": "❌ FAILURES",
                "value": fail_str,
                "inline": False,
            })
        
        # Failed samples
        if failed_samples:
            samples_str = "```\n"
            for i, sample in enumerate(failed_samples[:5], 1):
                text = sample.get('text', '')[:40]
                score = sample.get('score', 0)
                label = sample.get('label', 'FP')
                samples_str += f"{i}. \"{text}...\" → {score:.2f} ({label})\n"
            samples_str += "```"
            embed["fields"].append({
                "name": "🔍 FAILED SAMPLES (top 5)",
                "value": samples_str,
                "inline": False,
            })
        
        # Actions
        if actions:
            action_str = "\n".join([f"{i}. {a}" for i, a in enumerate(actions, 1)])
        else:
            action_str = "1. Do NOT deploy this model\n2. Add failed samples to adversarial training\n3. Investigate root cause\n4. Retrain with expanded data"
        
        embed["fields"].append({
            "name": "🛠️ ACTION REQUIRED",
            "value": action_str,
            "inline": False,
        })
        
        return self._send({"embeds": [embed]})
    
    def send_report(self, 
                    metrics: Dict[str, Any],
                    categories: Dict[str, Dict] = None,
                    trend: List[float] = None,
                    failed_samples: List[Dict] = None,
                    model_name: str = "payload_cnn") -> bool:
        """Send appropriate report based on metrics."""
        # Determine status
        failures = []
        warnings = []
        
        acc = metrics.get('accuracy', 0)
        fp = metrics.get('fp_rate', 0)
        fn = metrics.get('fn_rate', 0)
        lat = metrics.get('p95_latency_ms', 0)
        cal = metrics.get('calibration_error', 0)
        
        # Check failures
        if acc < self.config.accuracy_threshold:
            failures.append(f"Accuracy below threshold ({acc*100:.2f}% < {self.config.accuracy_threshold*100}%)")
        if fp > self.config.fp_threshold:
            failures.append(f"FP Rate exceeded ({fp*100:.2f}% > {self.config.fp_threshold*100}%)")
        if fn > self.config.fn_threshold:
            failures.append(f"FN Rate exceeded ({fn*100:.2f}% > {self.config.fn_threshold*100}%)")
        if lat > self.config.latency_threshold_ms:
            failures.append(f"Latency exceeded ({lat:.0f}ms > {self.config.latency_threshold_ms}ms)")
        if cal > self.config.calibration_threshold:
            failures.append(f"Calibration drift ({cal:.3f} > {self.config.calibration_threshold})")
        
        # Check warnings (approaching threshold)
        warn_ratio = self.config.warn_ratio
        if acc < self.config.accuracy_threshold / warn_ratio and acc >= self.config.accuracy_threshold:
            warnings.append(f"Accuracy approaching threshold ({acc*100:.2f}%)")
        if fp > self.config.fp_threshold * warn_ratio and fp <= self.config.fp_threshold:
            warnings.append(f"FP Rate approaching threshold ({fp*100:.2f}%)")
        if fn > self.config.fn_threshold * warn_ratio and fn <= self.config.fn_threshold:
            warnings.append(f"FN Rate approaching threshold ({fn*100:.2f}%)")
        if lat > self.config.latency_threshold_ms * warn_ratio and lat <= self.config.latency_threshold_ms:
            warnings.append(f"Latency approaching threshold ({lat:.0f}ms)")
        
        # Send appropriate message
        if failures:
            return self.send_failure(metrics, failures, failed_samples, model_name=model_name)
        elif warnings:
            return self.send_warning(metrics, warnings, categories, model_name=model_name)
        else:
            coverage = {cat: m.get('count', 0) for cat, m in (categories or {}).items()}
            return self.send_success(metrics, trend, coverage, model_name=model_name)


def send_discord_alert(metrics: Dict[str, Any], 
                       webhook_url: str = None,
                       **kwargs) -> bool:
    """Convenience function to send Discord alert."""
    alerter = DiscordAlerter(webhook_url)
    return alerter.send_report(metrics, **kwargs)


if __name__ == "__main__":
    # Test (won't actually send without webhook URL)
    alerter = DiscordAlerter()
    
    # Test success
    metrics = {
        'accuracy': 0.9987,
        'fp_rate': 0.012,
        'fn_rate': 0.008,
        'p95_latency_ms': 45,
        'calibration_error': 0.04,
    }
    
    print("Success message:")
    print(alerter._format_metrics_table(metrics))
    
    # Test failure
    metrics['fp_rate'] = 0.037
    metrics['accuracy'] = 0.982
    
    print("\nFailure metrics:")
    print(alerter._format_metrics_table(metrics))
