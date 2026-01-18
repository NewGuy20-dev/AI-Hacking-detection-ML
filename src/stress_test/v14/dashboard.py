"""HTML dashboard generator for V1.4 stress test results."""
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class DashboardGenerator:
    """Generate unified HTML dashboard from all model logs."""
    
    def __init__(self, logs_dir: Path, output_path: Path):
        self.logs_dir = Path(logs_dir)
        self.output_path = Path(output_path)
        
    def generate(self, run_date: str) -> Path:
        """Generate dashboard from all model logs for this run."""
        # Load all JSONL files
        all_results = self._load_all_logs(run_date)
        
        if not all_results:
            print("No log files found for this run")
            return None
        
        # Compute statistics
        stats = self._compute_stats(all_results)
        
        # Generate HTML
        html = self._render_html(stats, run_date)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding='utf-8')
        
        return self.output_path
    
    def _load_all_logs(self, run_date: str) -> Dict:
        """Load all JSONL logs for this run."""
        models = ['payload', 'url', 'timeseries', 'meta', 'fraud', 'host', 'network']
        all_results = {}
        
        for model in models:
            log_path = self.logs_dir / f"{model}_{run_date}.jsonl"
            if log_path.exists():
                results = []
                with open(log_path, encoding='utf-8') as f:
                    for line in f:
                        results.append(json.loads(line))
                all_results[model] = results
        
        return all_results
    
    def _compute_stats(self, all_results: Dict) -> Dict:
        """Compute aggregate statistics."""
        stats = {
            'models': {},
            'total_scenarios': 0,
            'total_passed': 0,
            'overall_accuracy': 0
        }
        
        for model, results in all_results.items():
            total = len(results)
            passed = sum(1 for r in results if r['passed'])
            failed = total - passed
            
            # Per-category stats
            categories = defaultdict(lambda: {'total': 0, 'passed': 0})
            for r in results:
                cat = r['category']
                categories[cat]['total'] += 1
                if r['passed']:
                    categories[cat]['passed'] += 1
            
            # Per-difficulty stats
            difficulties = defaultdict(lambda: {'total': 0, 'passed': 0})
            for r in results:
                diff = r.get('difficulty', 'medium')
                difficulties[diff]['total'] += 1
                if r['passed']:
                    difficulties[diff]['passed'] += 1
            
            # Failed samples
            failed_samples = [r for r in results if not r['passed']][:20]
            
            stats['models'][model] = {
                'total': total,
                'passed': passed,
                'failed': failed,
                'accuracy': passed / total if total > 0 else 0,
                'categories': dict(categories),
                'difficulties': dict(difficulties),
                'failed_samples': failed_samples
            }
            
            stats['total_scenarios'] += total
            stats['total_passed'] += passed
        
        stats['overall_accuracy'] = stats['total_passed'] / stats['total_scenarios'] if stats['total_scenarios'] > 0 else 0
        
        return stats
    
    def _render_html(self, stats: Dict, run_date: str) -> str:
        """Render HTML dashboard."""
        models_html = ""
        for model, data in stats['models'].items():
            acc = data['accuracy'] * 100
            status = "✅" if acc >= 95 else "⚠️" if acc >= 90 else "❌"
            
            # Category breakdown
            cat_rows = ""
            for cat, cat_data in data['categories'].items():
                cat_acc = (cat_data['passed'] / cat_data['total'] * 100) if cat_data['total'] > 0 else 0
                cat_rows += f"<tr><td>{cat}</td><td>{cat_acc:.1f}%</td><td>{cat_data['passed']}/{cat_data['total']}</td></tr>"
            
            # Difficulty breakdown
            diff_rows = ""
            for diff in ['easy', 'medium', 'hard', 'adversarial']:
                if diff in data.get('difficulties', {}):
                    diff_data = data['difficulties'][diff]
                    diff_acc = (diff_data['passed'] / diff_data['total'] * 100) if diff_data['total'] > 0 else 0
                    diff_rows += f"<tr><td>{diff.capitalize()}</td><td>{diff_acc:.1f}%</td><td>{diff_data['passed']}/{diff_data['total']}</td></tr>"
            
            # Failed samples
            failed_rows = ""
            for sample in data['failed_samples'][:10]:
                failed_rows += f"<tr><td>{sample['category']}</td><td>{sample['input_preview'][:50]}...</td><td>{sample['expected']}</td><td>{sample['predicted']}</td><td>{sample['confidence']:.2f}</td></tr>"
            
            models_html += f"""
            <div class="model-card">
                <h2>{status} {model.upper()}</h2>
                <div class="stats">
                    <div class="stat"><span class="label">Accuracy:</span> <span class="value">{acc:.1f}%</span></div>
                    <div class="stat"><span class="label">Total:</span> <span class="value">{data['total']:,}</span></div>
                    <div class="stat"><span class="label">Passed:</span> <span class="value pass">{data['passed']:,}</span></div>
                    <div class="stat"><span class="label">Failed:</span> <span class="value fail">{data['failed']:,}</span></div>
                </div>
                <h3>Difficulty Breakdown</h3>
                <table>
                    <tr><th>Difficulty</th><th>Accuracy</th><th>Passed/Total</th></tr>
                    {diff_rows}
                </table>
                <h3>Category Breakdown</h3>
                <table>
                    <tr><th>Category</th><th>Accuracy</th><th>Passed/Total</th></tr>
                    {cat_rows}
                </table>
                <h3>Failed Samples (Top 10)</h3>
                <table>
                    <tr><th>Category</th><th>Input</th><th>Expected</th><th>Got</th><th>Confidence</th></tr>
                    {failed_rows}
                </table>
            </div>
            """
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>V1.4 Stress Test Dashboard - {run_date}</title>
    <meta charset="UTF-8">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0a0e27; color: #e0e0e0; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 12px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header .summary {{ font-size: 1.2em; opacity: 0.9; }}
        .model-card {{ background: #1a1f3a; border-radius: 12px; padding: 25px; margin-bottom: 25px; border: 1px solid #2a2f4a; }}
        .model-card h2 {{ font-size: 1.8em; margin-bottom: 15px; color: #667eea; }}
        .model-card h3 {{ font-size: 1.3em; margin: 20px 0 10px 0; color: #a0a0a0; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat {{ background: #0f1429; padding: 15px; border-radius: 8px; }}
        .stat .label {{ display: block; font-size: 0.9em; color: #888; margin-bottom: 5px; }}
        .stat .value {{ display: block; font-size: 1.5em; font-weight: bold; }}
        .stat .value.pass {{ color: #4ade80; }}
        .stat .value.fail {{ color: #f87171; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #2a2f4a; }}
        th {{ background: #0f1429; color: #667eea; font-weight: 600; }}
        tr:hover {{ background: #0f1429; }}
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 V1.4 Stress Test Dashboard</h1>
        <div class="summary">
            Run Date: {run_date} | 
            Total Scenarios: {stats['total_scenarios']:,} | 
            Overall Accuracy: {stats['overall_accuracy']*100:.1f}%
        </div>
    </div>
    
    {models_html}
    
    <div class="footer">
        <p>Generated by V1.4 Stress Test Suite</p>
        <p>AI Hacking Detection ML System</p>
    </div>
</body>
</html>"""

