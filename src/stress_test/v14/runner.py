"""Adaptive scheduler and stress test runner for V1.4."""
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from scenarios import Scenario, ScenarioResult, ScenarioRegistry, PayloadGenerator, URLGenerator, TimeSeriesGenerator, TabularGenerator, MetaGenerator
from models import ModelWrapper
from logger import JSONLogger


# Risk-weighted base distributions
BASE_WEIGHTS = {
    'payload': {
        'sqli': 0.25, 'xss': 0.20, 'cmdi': 0.20, 'path_traversal': 0.15,
        'ssti': 0.10, 'xxe': 0.05, 'ldap': 0.05
    },
    'url': {
        'phishing': 0.30, 'typosquatting': 0.25, 'shorteners': 0.15,
        'homograph': 0.15, 'dga': 0.10, 'malware': 0.05
    },
    'timeseries': {
        'ddos': 0.30, 'portscan': 0.25, 'exfiltration': 0.20,
        'c2': 0.15, 'bruteforce': 0.10
    },
    'fraud': {
        'card_not_present': 0.40, 'account_takeover': 0.35, 'synthetic': 0.25
    },
    'host': {
        'spyware': 0.25, 'ransomware': 0.25, 'trojan': 0.20,
        'rootkit': 0.15, 'backdoor': 0.15
    },
    'network': {
        'dos': 0.35, 'probe': 0.30, 'r2l': 0.20, 'u2r': 0.15
    },
    'meta': {
        'combined': 1.0
    }
}


class AdaptiveScheduler:
    """Manages scenario distribution with adaptive weighting."""
    
    def __init__(self, base_weights: Dict[str, float], adaptive_ratio: float = 0.3):
        self.base_weights = base_weights
        self.adaptive_ratio = adaptive_ratio
        
    def compute_weights(self, category_accuracy: Dict[str, float]) -> Dict[str, float]:
        """Adjust weights based on model performance."""
        if not category_accuracy:
            return self.base_weights
        
        # Invert accuracy: lower accuracy = higher weight
        inv_acc = {cat: 1 - acc for cat, acc in category_accuracy.items()}
        inv_sum = sum(inv_acc.values()) or 1
        adaptive_weights = {cat: v / inv_sum for cat, v in inv_acc.items()}
        
        # Blend: 70% base + 30% adaptive
        final = {}
        all_cats = set(self.base_weights) | set(adaptive_weights)
        for cat in all_cats:
            base = self.base_weights.get(cat, 0)
            adapt = adaptive_weights.get(cat, 0)
            final[cat] = (1 - self.adaptive_ratio) * base + self.adaptive_ratio * adapt
        
        # Normalize
        total = sum(final.values())
        return {cat: w / total for cat, w in final.items()} if total > 0 else self.base_weights


class StressTestRunner:
    """Main runner for a single model."""
    
    def __init__(self, model_name: str, config: Dict):
        self.model_name = model_name
        self.target_duration_min = config.get('target_duration_min', 45)
        self.checkpoint_interval = config.get('checkpoint_interval', 500)
        self.models_dir = Path(config.get('models_dir', 'models'))
        self.scenarios_dir = Path(config.get('scenarios_dir', 'configs/stress_test/scenarios_v14'))
        self.output_dir = Path(config.get('output_dir', 'evaluation/stress_test_v14'))
        
    def run(self) -> Dict:
        """Run complete stress test for this model."""
        print(f"\n{'='*60}")
        print(f"  {self.model_name.upper()} STRESS TEST")
        print(f"{'='*60}")
        
        # Load model
        print(f"Loading {self.model_name} model...")
        model = ModelWrapper(self.model_name, self.models_dir).load()
        print(f"✓ Model loaded")
        
        # Load static scenarios
        print(f"Loading static scenarios...")
        registry = ScenarioRegistry(self.scenarios_dir)
        static_scenarios = registry.load_static(self.model_name)
        print(f"✓ Loaded {len(static_scenarios)} static scenarios")
        
        # Initialize generator and scheduler
        generator = self._get_generator()
        scheduler = AdaptiveScheduler(BASE_WEIGHTS.get(self.model_name, {}))
        
        # Run test
        run_date = datetime.now().strftime('%Y-%m-%d')
        with JSONLogger(self.output_dir, self.model_name, run_date) as logger:
            # Phase 1: Static scenarios
            if static_scenarios:
                print(f"\nPhase 1: Running static scenarios...")
                for scenario in tqdm(static_scenarios, desc="Static"):
                    result = self._run_scenario(model, scenario)
                    logger.log(result)
                
                print(f"✓ Static phase complete")
                print(f"  Accuracy: {logger.get_summary()['accuracy']*100:.1f}%")
            
            # Phase 2: Dynamic scenarios
            if generator:
                print(f"\nPhase 2: Running dynamic scenarios (target: {self.target_duration_min} min)...")
                start_time = time.time()
                dynamic_count = 0
                
                pbar = tqdm(desc="Dynamic", unit=" scenarios")
                while (time.time() - start_time) / 60 < self.target_duration_min:
                    # Get adaptive weights
                    weights = scheduler.compute_weights(logger.get_category_accuracy())
                    
                    # Generate batch
                    if self.model_name in ['fraud', 'host', 'network']:
                        batch = generator.generate(self.model_name, 100, weights)
                    else:
                        batch = generator.generate(100, weights)
                    
                    for scenario in batch:
                        result = self._run_scenario(model, scenario)
                        logger.log(result)
                        dynamic_count += 1
                        pbar.update(1)
                        
                        if dynamic_count % self.checkpoint_interval == 0:
                            elapsed = (time.time() - start_time) / 60
                            acc = logger.get_summary()['accuracy']
                            pbar.set_postfix({
                                'elapsed': f'{elapsed:.1f}m',
                                'acc': f'{acc*100:.1f}%'
                            })
                
                pbar.close()
            else:
                dynamic_count = 0
            
            # Final summary
            summary = logger.get_summary()
            total_duration = (time.time() - start_time) / 60 if generator else 0
            
            print(f"\n✓ Test complete!")
            print(f"  Static: {len(static_scenarios)} scenarios")
            print(f"  Dynamic: {dynamic_count} scenarios")
            print(f"  Total: {summary['total_scenarios']} scenarios")
            print(f"  Duration: {total_duration:.1f} min")
            print(f"  Accuracy: {summary['accuracy']*100:.1f}%")
            print(f"  Passed: {summary['passed']}/{summary['total_scenarios']}")
            
            # Display per-difficulty accuracy
            if summary.get('accuracy_by_difficulty'):
                print(f"\n  Accuracy by Difficulty:")
                for diff in ['easy', 'medium', 'hard', 'adversarial']:
                    if diff in summary['accuracy_by_difficulty']:
                        acc = summary['accuracy_by_difficulty'][diff] * 100
                        stats = summary['difficulty_breakdown'][diff]
                        print(f"    {diff.capitalize():12s}: {acc:5.1f}% ({stats['passed']}/{stats['total']})")
            
            return {
                'model': self.model_name,
                'static_count': len(static_scenarios),
                'dynamic_count': dynamic_count,
                'total_scenarios': summary['total_scenarios'],
                'total_duration_min': total_duration,
                'accuracy': summary['accuracy'],
                'accuracy_by_difficulty': summary.get('accuracy_by_difficulty', {}),
                'difficulty_breakdown': summary.get('difficulty_breakdown', {}),
                'final_stats': summary['categories']
            }
    
    def _get_generator(self):
        """Get appropriate generator for this model."""
        if self.model_name == 'payload':
            return PayloadGenerator()
        elif self.model_name == 'url':
            return URLGenerator()
        elif self.model_name == 'timeseries':
            return TimeSeriesGenerator()
        elif self.model_name in ['fraud', 'host', 'network']:
            return TabularGenerator()
        elif self.model_name == 'meta':
            return MetaGenerator()
        else:
            return None
    
    def _run_scenario(self, model: ModelWrapper, scenario: Scenario) -> ScenarioResult:
        """Run a single scenario."""
        try:
            pred, conf, latency = model.predict(scenario.input_data)
            passed = (pred == scenario.expected_label)
            
            return ScenarioResult(
                scenario=scenario,
                prediction=pred,
                confidence=conf,
                passed=passed,
                latency_ms=latency,
                timestamp=datetime.now().isoformat(),
                error=None
            )
        except Exception as e:
            return ScenarioResult(
                scenario=scenario,
                prediction=-1,
                confidence=0.0,
                passed=False,
                latency_ms=0.0,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )

