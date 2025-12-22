"""Claymorphism UI Dashboard for AI Hacking Detection."""
import gradio as gr
import torch
import numpy as np
from pathlib import Path
import csv
import tempfile
import json
from datetime import datetime

# Global state
predictor = None
scan_history = []
stats = {"total": 0, "malicious": 0, "safe": 0, "by_type": {}}

CSS_PATH = Path(__file__).parent / "static" / "claymorphism.css"
CSS = CSS_PATH.read_text() if CSS_PATH.exists() else ""

def load_models():
    global predictor
    from src.hybrid_predictor import HybridPredictor
    predictor = HybridPredictor('models')
    predictor.load_models()

def classify_attack_type(text: str, confidence: float) -> tuple[str, str]:
    """Return (type, icon)."""
    if confidence < 0.75:
        return "BENIGN", "✅"
    t = text.lower()
    if any(p in t for p in ["'", "or ", "union", "select", "--"]):
        return "SQL Injection", "💉"
    if any(p in t for p in ["<script", "onerror", "javascript:"]):
        return "XSS", "🔗"
    if any(p in t for p in [";", "|", "`", "$("]):
        return "Command Injection", "⚡"
    if any(p in t for p in ["{{", "${", "<%"]):
        return "Template Injection", "📝"
    return "Unknown Attack", "⚠️"

def get_severity(conf: float) -> tuple[str, str, str]:
    """Return (label, emoji, css_class)."""
    if conf > 0.95: return "CRITICAL", "🔴", "critical"
    if conf > 0.85: return "HIGH", "🟠", "high"
    if conf > 0.7: return "MEDIUM", "🟡", "medium"
    return "LOW", "🟢", "low"

def format_result_html(verdict: str, confidence: float, attack_type: str, icon: str, severity: tuple, extra: dict = None) -> str:
    """Generate styled result card HTML."""
    sev_label, sev_emoji, sev_class = severity
    is_mal = verdict == "MALICIOUS"
    card_class = "malicious" if is_mal else "safe"
    conf_class = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"
    
    scores_html = ""
    if extra and "scores" in extra:
        scores_html = "<div style='margin-top:12px;'><strong>Model Scores:</strong><br>"
        for k, v in extra["scores"].items():
            scores_html += f"<span style='margin-right:12px;'>{k}: {v:.1%}</span>"
        scores_html += "</div>"
    
    return f"""
    <div class="result-card {card_class}" role="region" aria-label="Scan Result">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px;">
            <div>
                <div style="font-size:1.5rem;font-weight:700;margin-bottom:8px;">
                    {icon} {verdict}
                </div>
                <div style="color:var(--clay-text-muted);margin-bottom:4px;">
                    <strong>Attack Type:</strong> {attack_type}
                </div>
            </div>
            <div class="severity-badge severity-{sev_class}" role="status" aria-label="Severity: {sev_label}">
                {sev_emoji} {sev_label}
            </div>
        </div>
        <div style="margin-top:16px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                <span><strong>Confidence</strong></span>
                <span>{confidence:.1%}</span>
            </div>
            <div class="confidence-meter" role="progressbar" aria-valuenow="{int(confidence*100)}" aria-valuemin="0" aria-valuemax="100">
                <div class="confidence-fill {conf_class}" style="width:{confidence*100}%"></div>
            </div>
        </div>
        {scores_html}
    </div>
    """

def update_stats(is_attack: bool, attack_type: str):
    global stats
    stats["total"] += 1
    if is_attack:
        stats["malicious"] += 1
        stats["by_type"][attack_type] = stats["by_type"].get(attack_type, 0) + 1
    else:
        stats["safe"] += 1

def add_to_history(input_text: str, verdict: str, confidence: float, scan_type: str):
    global scan_history
    scan_history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "input": input_text[:40] + "..." if len(input_text) > 40 else input_text,
        "verdict": verdict,
        "confidence": f"{confidence:.1%}",
        "type": scan_type
    })
    scan_history = scan_history[:50]  # Keep last 50

def scan_payload(text: str):
    """Analyze payload with styled output."""
    if not text or not text.strip():
        return """<div class="result-card" role="alert">
            <p style="color:var(--clay-text-muted);">⌨️ Enter a payload to analyze</p>
        </div>"""
    
    if predictor is None:
        load_models()
    
    result = predictor.predict({'payloads': [text]})
    conf = float(result['confidence'][0])
    payload_score = float(result['scores']['payload'][0])
    
    # Use payload-specific score with higher threshold to reduce false positives
    # Short benign messages often get inflated ensemble scores
    is_attack = payload_score > 0.75
    
    attack_type, icon = classify_attack_type(text, payload_score)
    severity = get_severity(payload_score) if is_attack else ("LOW", "🟢", "low")
    verdict = "MALICIOUS" if is_attack else "SAFE"
    
    update_stats(is_attack, attack_type)
    add_to_history(text, verdict, payload_score, "Payload")
    
    return format_result_html(
        verdict, payload_score, attack_type, icon, severity,
        {"scores": {k: float(v[0]) for k, v in result['scores'].items()}}
    )

def scan_url(url: str):
    """Analyze URL with styled output."""
    if not url or not url.strip():
        return """<div class="result-card" role="alert">
            <p style="color:var(--clay-text-muted);">🔗 Enter a URL to analyze</p>
        </div>"""
    
    if predictor is None:
        load_models()
    
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
    
    result = predictor.predict({'urls': [url]})
    conf = float(result['confidence'][0])
    is_attack = bool(result['is_attack'][0])
    severity = get_severity(conf)
    verdict = "MALICIOUS" if is_attack else "SAFE"
    attack_type = "Malicious URL" if is_attack else "Safe URL"
    icon = "🚫" if is_attack else "✅"
    
    update_stats(is_attack, attack_type)
    add_to_history(url, verdict, conf, "URL")
    
    extra_html = f"""
    <div style="margin-top:12px;padding:12px;background:var(--clay-bg);border-radius:var(--clay-radius-sm);">
        <strong>URL:</strong> <code style="word-break:break-all;">{url}</code>
    </div>
    <div style="margin-top:8px;">
        <strong>Recommendation:</strong> {"🚫 Block this URL" if conf > 0.7 else "✅ Allow"}
    </div>
    """
    
    return format_result_html(verdict, conf, attack_type, icon, severity) + extra_html

def batch_scan(file, progress=gr.Progress()):
    """Process batch with progress indicator."""
    if file is None:
        return None, None, """<div class="result-card">📁 Upload a file to begin batch analysis</div>"""
    
    if predictor is None:
        load_models()
    
    content = Path(file.name).read_text(errors='ignore')
    lines = [l.strip() for l in content.splitlines() if l.strip()][:100]
    
    if not lines:
        return None, None, """<div class="result-card">⚠️ File is empty or invalid</div>"""
    
    results = []
    mal_count = 0
    
    for i, line in enumerate(progress.tqdm(lines, desc="Scanning...")):
        result = predictor.predict({'payloads': [line]})
        is_attack = result['is_attack'][0]
        conf = float(result['confidence'][0])
        if is_attack:
            mal_count += 1
        results.append({
            'Input': line[:50] + '...' if len(line) > 50 else line,
            'Verdict': '⚠️ Malicious' if is_attack else '✅ Safe',
            'Confidence': f"{conf:.1%}"
        })
    
    # Create CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=['Input', 'Verdict', 'Confidence'])
        writer.writeheader()
        writer.writerows(results)
        output_path = f.name
    
    summary = f"""
    <div class="result-card">
        <div style="font-size:1.25rem;font-weight:700;margin-bottom:12px;">📊 Batch Analysis Complete</div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;">
            <div class="stat-card" style="flex:1;min-width:100px;">
                <div class="stat-value">{len(lines)}</div>
                <div class="stat-label">Total Scanned</div>
            </div>
            <div class="stat-card" style="flex:1;min-width:100px;border-left:3px solid var(--clay-danger);">
                <div class="stat-value" style="color:var(--clay-danger-dark);">{mal_count}</div>
                <div class="stat-label">Malicious</div>
            </div>
            <div class="stat-card" style="flex:1;min-width:100px;border-left:3px solid var(--clay-success);">
                <div class="stat-value" style="color:#38a169;">{len(lines) - mal_count}</div>
                <div class="stat-label">Safe</div>
            </div>
        </div>
    </div>
    """
    
    return results, output_path, summary

def get_history_html():
    """Generate history list HTML."""
    if not scan_history:
        return """<div class="result-card" style="text-align:center;">
            <p style="color:var(--clay-text-muted);">📜 No scan history yet</p>
        </div>"""
    
    items = ""
    for h in scan_history[:20]:
        verdict_color = "var(--clay-danger-dark)" if h["verdict"] == "MALICIOUS" else "var(--clay-success)"
        items += f"""
        <div class="history-item">
            <div style="flex:1;">
                <div style="font-weight:500;">{h['input']}</div>
                <div style="font-size:0.8rem;color:var(--clay-text-muted);">{h['type']} • {h['time']}</div>
            </div>
            <div style="text-align:right;">
                <div style="color:{verdict_color};font-weight:600;">{h['verdict']}</div>
                <div style="font-size:0.8rem;">{h['confidence']}</div>
            </div>
        </div>
        """
    
    return f"""<div role="list" aria-label="Scan History">{items}</div>"""

def export_history():
    """Export history to JSON."""
    if not scan_history:
        return None
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(scan_history, f, indent=2)
        return f.name

def get_stats_html():
    """Generate dashboard stats HTML."""
    type_items = ""
    for t, c in sorted(stats["by_type"].items(), key=lambda x: -x[1])[:5]:
        type_items += f"""
        <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid var(--clay-border);">
            <span>{t}</span><span style="font-weight:600;">{c}</span>
        </div>
        """
    
    if not type_items:
        type_items = "<p style='color:var(--clay-text-muted);'>No attacks detected yet</p>"
    
    return f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-bottom:24px;">
        <div class="stat-card">
            <div class="stat-value">{stats['total']}</div>
            <div class="stat-label">Total Scans</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:var(--clay-danger-dark);">{stats['malicious']}</div>
            <div class="stat-label">Threats Found</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color:#38a169;">{stats['safe']}</div>
            <div class="stat-label">Safe Inputs</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{stats['malicious']/max(stats['total'],1)*100:.0f}%</div>
            <div class="stat-label">Threat Rate</div>
        </div>
    </div>
    <div class="clay-card">
        <div style="font-weight:600;margin-bottom:12px;">🎯 Attack Type Distribution</div>
        {type_items}
    </div>
    """

def get_model_info_html():
    """Return model info as styled HTML."""
    if predictor is None:
        load_models()
    
    models_html = ""
    for name in predictor.pytorch_models.keys():
        models_html += f'<span class="severity-badge severity-low" style="margin:4px;">{name}</span>'
    for name in predictor.sklearn_models.keys():
        models_html += f'<span class="severity-badge severity-medium" style="margin:4px;">{name}</span>'
    
    return f"""
    <div class="clay-card" style="margin-bottom:16px;">
        <div style="font-weight:600;margin-bottom:12px;">🧠 Loaded Models</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px;">{models_html}</div>
        <div style="margin-top:12px;color:var(--clay-text-muted);">
            Device: <strong>{predictor.device}</strong>
        </div>
    </div>
    <div class="clay-card">
        <div style="font-weight:600;margin-bottom:12px;">📈 Model Performance</div>
        <div style="display:grid;gap:12px;">
            <div style="display:flex;justify-content:space-between;">
                <span>Payload CNN</span>
                <span style="font-weight:600;color:#38a169;">99.89%</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span>URL CNN</span>
                <span style="font-weight:600;color:#38a169;">97.47%</span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span>TimeSeries LSTM</span>
                <span style="font-weight:600;color:var(--clay-warning);">75.38%</span>
            </div>
        </div>
    </div>
    """

# Build UI
with gr.Blocks(title="AI Hacking Detection", css=CSS, theme=gr.themes.Soft()) as demo:
    
    # Header with theme toggle
    gr.HTML("""
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <div class="app-header">
        <div class="app-title">🛡️ AI Hacking Detection</div>
        <div class="theme-toggle" role="group" aria-label="Theme switcher">
            <button onclick="document.body.removeAttribute('data-theme')" class="active" aria-label="Light theme">☀️</button>
            <button onclick="document.body.setAttribute('data-theme','dark')" aria-label="Dark theme">🌙</button>
        </div>
    </div>
    """)
    
    gr.HTML('<div id="main-content"></div>')
    
    with gr.Tabs() as tabs:
        # Payload Scanner
        with gr.TabItem("🔍 Payload", id="payload"):
            gr.Markdown("### Analyze text payloads for attacks")
            payload_input = gr.Textbox(
                label="Payload",
                placeholder="Enter payload (e.g., ' OR 1=1--)",
                lines=3,
                elem_classes=["clay-input"]
            )
            payload_btn = gr.Button("🔍 Analyze", variant="primary", elem_classes=["clay-btn-primary"])
            payload_output = gr.HTML(
                value='<div class="result-card"><p style="color:var(--clay-text-muted);">⌨️ Enter a payload to analyze</p></div>'
            )
            gr.Examples(
                [["' OR '1'='1"], ["<script>alert('XSS')</script>"], ["; cat /etc/passwd"], ["Hello world"]],
                inputs=payload_input,
                label="Try these examples"
            )
        
        # URL Scanner
        with gr.TabItem("🌐 URL", id="url"):
            gr.Markdown("### Analyze URLs for malicious content")
            url_input = gr.Textbox(
                label="URL",
                placeholder="https://example.com",
                elem_classes=["clay-input"]
            )
            url_btn = gr.Button("🔍 Analyze", variant="primary", elem_classes=["clay-btn-primary"])
            url_output = gr.HTML(
                value='<div class="result-card"><p style="color:var(--clay-text-muted);">🔗 Enter a URL to analyze</p></div>'
            )
            gr.Examples(
                [["https://google.com"], ["http://paypa1-secure.tk/login"], ["https://github.com"]],
                inputs=url_input
            )
        
        # Batch Analysis
        with gr.TabItem("📁 Batch", id="batch"):
            gr.Markdown("### Upload file with multiple inputs (one per line)")
            file_input = gr.File(label="Upload TXT file", file_types=[".txt"], elem_classes=["file-upload"])
            batch_btn = gr.Button("🚀 Process Batch", variant="primary", elem_classes=["clay-btn-primary"])
            batch_summary = gr.HTML(value='<div class="result-card">📁 Upload a file to begin</div>')
            batch_output = gr.Dataframe(label="Results", headers=["Input", "Verdict", "Confidence"])
            download_btn = gr.File(label="📥 Download CSV")
        
        # History
        with gr.TabItem("📜 History", id="history"):
            gr.Markdown("### Recent Scan History")
            refresh_hist_btn = gr.Button("🔄 Refresh", elem_classes=["clay-btn"])
            history_output = gr.HTML(value=get_history_html())
            export_btn = gr.Button("📥 Export History", elem_classes=["clay-btn"])
            export_file = gr.File(label="Download")
        
        # Dashboard
        with gr.TabItem("📊 Dashboard", id="dashboard"):
            gr.Markdown("### Statistics & Analytics")
            refresh_stats_btn = gr.Button("🔄 Refresh Stats", elem_classes=["clay-btn"])
            stats_output = gr.HTML(value=get_stats_html())
        
        # Model Info
        with gr.TabItem("🧠 Models", id="models"):
            gr.Markdown("### Model Information")
            info_btn = gr.Button("🔄 Load Info", elem_classes=["clay-btn"])
            info_output = gr.HTML()
            gr.Markdown("### ⚠️ Known Limitations")
            gr.Dataframe(
                value=[
                    ["<3 emoji", "~95% FP", "< resembles HTML"],
                    ["SELECT FROM menu", "~72% flagged", "SQL pattern"],
                ],
                headers=["Pattern", "Behavior", "Reason"]
            )
        
        # About
        with gr.TabItem("ℹ️ About", id="about"):
            gr.HTML("""
            <div class="clay-card">
                <h2 style="margin-top:0;">AI Hacking Detection System</h2>
                <p>Enterprise-grade threat detection using ensemble ML models.</p>
                <h3>🎯 Detection Capabilities</h3>
                <ul>
                    <li>SQL Injection</li>
                    <li>Cross-Site Scripting (XSS)</li>
                    <li>Command Injection</li>
                    <li>Malicious URLs & Phishing</li>
                </ul>
                <h3>🔧 Technology</h3>
                <ul>
                    <li>PyTorch Deep Learning</li>
                    <li>Character-level CNNs</li>
                    <li>LSTM for time-series</li>
                    <li>Ensemble Meta-classifier</li>
                </ul>
                <p style="margin-top:24px;color:var(--clay-text-muted);">
                    <strong>License:</strong> MIT
                </p>
            </div>
            """)
    
    # Event handlers
    payload_btn.click(scan_payload, inputs=payload_input, outputs=payload_output)
    payload_input.submit(scan_payload, inputs=payload_input, outputs=payload_output)
    
    url_btn.click(scan_url, inputs=url_input, outputs=url_output)
    url_input.submit(scan_url, inputs=url_input, outputs=url_output)
    
    batch_btn.click(batch_scan, inputs=file_input, outputs=[batch_output, download_btn, batch_summary])
    
    refresh_hist_btn.click(get_history_html, outputs=history_output)
    export_btn.click(export_history, outputs=export_file)
    
    refresh_stats_btn.click(get_stats_html, outputs=stats_output)
    info_btn.click(get_model_info_html, outputs=info_output)

if __name__ == "__main__":
    load_models()
    demo.launch()
