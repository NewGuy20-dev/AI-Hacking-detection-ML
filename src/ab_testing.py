"""A/B Testing framework for model comparison."""
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.model_registry import ModelRegistry
from src.model_monitor import ModelMonitor


class ABTestManager:
    """Manage A/B testing experiments for model comparison."""
    
    def __init__(self, registry: ModelRegistry = None, monitor: ModelMonitor = None,
                 storage_path: str = "evaluation/ab_tests"):
        self.registry = registry or ModelRegistry('models')
        self.monitor = monitor or ModelMonitor()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, dict] = {}
        self._load_experiments()
    
    def create_experiment(self, name: str, model_a: dict, model_b: dict,
                         split_ratio: float = 0.5, min_samples: int = 1000) -> dict:
        """Create a new A/B test experiment."""
        if name in self._experiments:
            raise ValueError(f"Experiment '{name}' already exists")
        
        experiment = {
            "name": name,
            "status": "running",
            "created_at": datetime.now().isoformat(),
            "model_a": model_a,
            "model_b": model_b,
            "split_ratio": split_ratio,
            "min_samples": min_samples,
            "results": {
                "A": {"samples": 0, "predictions": [], "latencies": [], "correct": 0},
                "B": {"samples": 0, "predictions": [], "latencies": [], "correct": 0}
            }
        }
        
        self._experiments[name] = experiment
        self._save_experiment(name)
        return experiment
    
    def get_model(self, experiment_name: str, model_class=None) -> Tuple[any, str]:
        """Get model for prediction based on split ratio."""
        exp = self._experiments.get(experiment_name)
        if not exp or exp['status'] != 'running':
            raise ValueError(f"Experiment '{experiment_name}' not found or not running")
        
        variant = "B" if random.random() < exp['split_ratio'] else "A"
        model_info = exp[f'model_{variant.lower()}']
        
        model = self.registry.load(model_info['name'], model_info.get('version'), model_class)
        return model, variant
    
    def record_outcome(self, experiment_name: str, variant: str,
                       prediction: float, actual: int = None, latency_ms: float = None):
        """Record prediction outcome for analysis."""
        exp = self._experiments.get(experiment_name)
        if not exp:
            return
        
        results = exp['results'][variant]
        results['samples'] += 1
        results['predictions'].append(float(prediction))
        
        if latency_ms is not None:
            results['latencies'].append(latency_ms)
        
        if actual is not None:
            predicted_class = 1 if prediction > 0.5 else 0
            if predicted_class == actual:
                results['correct'] += 1
        
        # Keep only last 10000 predictions to save memory
        if len(results['predictions']) > 10000:
            results['predictions'] = results['predictions'][-10000:]
            results['latencies'] = results['latencies'][-10000:]
        
        self._save_experiment(experiment_name)
    
    def get_results(self, experiment_name: str) -> dict:
        """Get current results for an experiment."""
        exp = self._experiments.get(experiment_name)
        if not exp:
            return {"error": "Experiment not found"}
        
        results_a = exp['results']['A']
        results_b = exp['results']['B']
        
        metrics = {}
        for variant, results in [('A', results_a), ('B', results_b)]:
            if results['samples'] > 0:
                metrics[variant] = {
                    'samples': results['samples'],
                    'accuracy': results['correct'] / results['samples'] if results['samples'] > 0 else None,
                    'mean_confidence': np.mean(results['predictions']) if results['predictions'] else None,
                    'latency_p50': np.percentile(results['latencies'], 50) if len(results['latencies']) >= 10 else None,
                    'latency_p95': np.percentile(results['latencies'], 95) if len(results['latencies']) >= 20 else None
                }
        
        total_samples = results_a['samples'] + results_b['samples']
        significance = self._calculate_significance(results_a, results_b)
        
        return {
            "experiment": experiment_name,
            "status": exp['status'],
            "samples": {"A": results_a['samples'], "B": results_b['samples']},
            "metrics": metrics,
            "winner": significance.get('winner'),
            "p_value": significance.get('p_value'),
            "significant": significance.get('significant', False),
            "recommendation": self._get_recommendation(exp, significance, total_samples)
        }
    
    def determine_winner(self, experiment_name: str) -> dict:
        """Determine if there's a statistically significant winner."""
        exp = self._experiments.get(experiment_name)
        if not exp:
            return {"error": "Experiment not found"}
        
        return self._calculate_significance(exp['results']['A'], exp['results']['B'])
    
    def end_experiment(self, experiment_name: str, deploy_winner: bool = False) -> dict:
        """End experiment and optionally deploy winner."""
        exp = self._experiments.get(experiment_name)
        if not exp:
            return {"error": "Experiment not found"}
        
        exp['status'] = 'completed'
        exp['ended_at'] = datetime.now().isoformat()
        
        results = self.get_results(experiment_name)
        
        if deploy_winner and results.get('winner'):
            winner_model = exp[f"model_{results['winner'].lower()}"]
            self.registry.set_active(winner_model['name'], winner_model.get('version'))
            results['deployed'] = True
        
        self._save_experiment(experiment_name)
        return results
    
    def list_experiments(self, status: str = None) -> List[dict]:
        """List all experiments, optionally filtered by status."""
        experiments = []
        for name, exp in self._experiments.items():
            if status is None or exp['status'] == status:
                experiments.append({
                    'name': name,
                    'status': exp['status'],
                    'created_at': exp['created_at'],
                    'samples': exp['results']['A']['samples'] + exp['results']['B']['samples']
                })
        return experiments
    
    def _calculate_significance(self, results_a: dict, results_b: dict) -> dict:
        """Calculate statistical significance."""
        if results_a['samples'] < 30 or results_b['samples'] < 30:
            return {'significant': False, 'reason': 'insufficient_samples'}
        
        # Use accuracy if available, otherwise use mean prediction
        if results_a['correct'] > 0 and results_b['correct'] > 0:
            acc_a = results_a['correct'] / results_a['samples']
            acc_b = results_b['correct'] / results_b['samples']
            
            # Simple z-test for proportions
            p_pooled = (results_a['correct'] + results_b['correct']) / (results_a['samples'] + results_b['samples'])
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/results_a['samples'] + 1/results_b['samples']))
            
            if se > 0:
                z = (acc_b - acc_a) / se
                # Two-tailed p-value approximation
                p_value = 2 * (1 - min(0.9999, abs(z) / 4 + 0.5))
            else:
                p_value = 1.0
            
            winner = None
            if p_value < 0.05:
                winner = 'B' if acc_b > acc_a else 'A'
            
            return {
                'significant': p_value < 0.05,
                'p_value': round(p_value, 4),
                'winner': winner,
                'improvement': round(abs(acc_b - acc_a), 4) if winner else None
            }
        
        return {'significant': False, 'reason': 'no_ground_truth'}
    
    def _get_recommendation(self, exp: dict, significance: dict, total_samples: int) -> str:
        if exp['status'] == 'completed':
            return "Experiment completed"
        if significance.get('significant') and significance.get('winner'):
            return f"Deploy model {significance['winner']} - statistically significant improvement"
        if total_samples < exp['min_samples']:
            remaining = exp['min_samples'] - total_samples
            return f"Continue collecting data ({remaining} more samples needed)"
        return "No significant difference detected"
    
    def _save_experiment(self, name: str):
        """Persist experiment state to JSON file."""
        exp = self._experiments.get(name)
        if exp:
            path = self.storage_path / f"{name}.json"
            with open(path, 'w') as f:
                json.dump(exp, f, indent=2)
    
    def _load_experiments(self):
        """Load all experiment states from storage."""
        for path in self.storage_path.glob("*.json"):
            try:
                with open(path) as f:
                    exp = json.load(f)
                    self._experiments[exp['name']] = exp
            except (json.JSONDecodeError, KeyError):
                continue
