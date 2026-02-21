#!/usr/bin/env python3
"""V1.4 Comprehensive Stress Test Suite - CLI Entry Point."""
import argparse
import sys
from pathlib import Path
from datetime import date
from typing import Any, Optional, Tuple, Type

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Imported lazily so --help works even when runtime ML deps are not installed.
StressTestRunner = None
DashboardGenerator = None

DEFAULT_MODELS = ['payload', 'url', 'timeseries', 'meta', 'fraud', 'host', 'network']
SUPPORTED_MODELS = [*DEFAULT_MODELS, 'anomaly']


class TeeLogger:
    """Write to both stdout and file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def _resolve_runtime_components(require_dashboard: bool = True) -> Tuple[Optional[Type[Any]], Optional[Type[Any]], Optional[str]]:
    """Load heavy runtime components with a user-friendly dependency error."""
    runner_cls = globals().get("StressTestRunner")
    dashboard_cls = globals().get("DashboardGenerator")

    if runner_cls is None:
        try:
            from stress_test.v14.runner import StressTestRunner as _runner
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            return None, None, (
                f"Missing dependency '{missing}'. Install project dependencies first "
                f"(for example: py -3 -m pip install -r requirements.txt on Windows)."
            )
        globals()["StressTestRunner"] = _runner
        runner_cls = _runner

    if require_dashboard and dashboard_cls is None:
        try:
            from stress_test.v14.dashboard import DashboardGenerator as _dashboard
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            return runner_cls, None, (
                f"Missing dependency '{missing}'. Install project dependencies first "
                f"(for example: py -3 -m pip install -r requirements.txt on Windows)."
            )
        globals()["DashboardGenerator"] = _dashboard
        dashboard_cls = _dashboard

    return runner_cls, dashboard_cls, None


def main():
    parser = argparse.ArgumentParser(
        description='V1.4 Comprehensive Stress Test Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run all models (default)
  python src/stress_test/stress_test_v14.py
  
  # Run specific models
  python src/stress_test/stress_test_v14.py --model url,payload
  
  # Quick test (5 min per model)
  python src/stress_test/stress_test_v14.py --duration 5
  
  # Skip dashboard generation
  python src/stress_test/stress_test_v14.py --no-dashboard
        '''
    )
    parser.add_argument('--model', type=str, default='all',
                        help='Models to test: all, or comma-separated (e.g., url,payload)')
    parser.add_argument('--duration', type=int, default=45,
                        help='Target duration per model in minutes (default: 45)')
    parser.add_argument('--output-dir', type=str, default='evaluation/stress_test_v14',
                        help='Output directory for logs and dashboard')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--scenarios-dir', type=str, default='configs/stress_test/scenarios_v14',
                        help='Directory containing scenario YAML files')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Skip dashboard generation')
    parser.add_argument('--checkpoint-interval', type=int, default=500,
                        help='Log progress every N scenarios (default: 500)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for deterministic dynamic scenario generation')
    args = parser.parse_args()

    runner_cls, dashboard_cls, import_error = _resolve_runtime_components(require_dashboard=not args.no_dashboard)
    if import_error:
        print(f"ERROR: {import_error}")
        sys.exit(2)
    if runner_cls is None or (not args.no_dashboard and dashboard_cls is None):
        print("ERROR: Failed to resolve runtime components.")
        sys.exit(2)
    assert runner_cls is not None
    
    # Setup dual logging to terminal and test.log
    log_file = Path('test.log')
    tee = TeeLogger(log_file)
    sys.stdout = tee
    
    try:
        # Parse model selection
        if args.model == 'all':
            models = DEFAULT_MODELS
        else:
            models = [m.strip() for m in args.model.split(',')]
            invalid = set(models) - set(SUPPORTED_MODELS)
            if invalid:
                print(f"❌ Invalid models: {invalid}")
                print(f"   Valid models: {', '.join(SUPPORTED_MODELS)}")
                sys.exit(1)
        
        output_dir = Path(args.output_dir)
        run_date = date.today().isoformat()
        
        # Print header
        print("=" * 70)
        print("  V1.4 COMPREHENSIVE STRESS TEST SUITE")
        print("=" * 70)
        print(f"  Models: {', '.join(models)}")
        print(f"  Target: {args.duration} min/model")
        print(f"  Output: {output_dir}")
        print(f"  Date: {run_date}")
        print("=" * 70)
        
        # Run tests
        results = {}
        for i, model in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Testing {model.upper()}...")
            
            config = {
                'target_duration_min': args.duration,
                'checkpoint_interval': args.checkpoint_interval,
                'models_dir': args.models_dir,
                'scenarios_dir': args.scenarios_dir,
                'output_dir': args.output_dir,
                'seed': args.seed,
            }
            
            try:
                runner = runner_cls(model, config)
                results[model] = runner.run()
            except Exception as e:
                print(f"❌ Error testing {model}: {e}")
                results[model] = {
                    'model': model,
                    'error': str(e),
                    'total_scenarios': 0,
                    'accuracy': 0
                }
        
        # Generate unified dashboard
        if not args.no_dashboard:
            print(f"\n{'='*70}")
            print("  GENERATING UNIFIED DASHBOARD")
            print(f"{'='*70}")
            
            dashboard_path = output_dir / f"dashboard_{run_date}.html"
            try:
                generator = dashboard_cls(output_dir, dashboard_path)
                generator.generate(run_date)
                
                print(f"✓ Dashboard: {dashboard_path}")
                print(f"  Open in browser: file://{dashboard_path.absolute()}")
            except Exception as e:
                print(f"❌ Dashboard generation failed: {e}")
        
        # Final summary
        print(f"\n{'='*70}")
        print("  FINAL SUMMARY")
        print(f"{'='*70}")
        
        total_scenarios = sum(r.get('total_scenarios', 0) for r in results.values())
        total_time = sum(r.get('total_duration_min', 0) for r in results.values())
        
        print(f"  Total Scenarios: {total_scenarios:,}")
        print(f"  Total Time: {total_time:.1f} min ({total_time/60:.1f} hours)")
        print()
        
        # Per-model summary
        for model, r in results.items():
            if 'error' in r:
                print(f"  ❌ {model:12s}: ERROR - {r['error']}")
            else:
                acc = r.get('accuracy', 0)
                status = "✅" if acc >= 0.95 else "⚠️" if acc >= 0.90 else "❌"
                print(f"  {status} {model:12s}: {acc*100:5.1f}% accuracy "
                      f"({r.get('total_scenarios', 0):,} scenarios, {r.get('total_duration_min', 0):.1f} min)")
        
        print(f"\n{'='*70}")
        
        # Exit code based on results
        successful = [r for r in results.values() if 'error' not in r]
        has_errors = any('error' in r for r in results.values())
        all_passed = bool(successful) and not has_errors and all(r.get('accuracy', 0) >= 0.90 for r in successful)
    
    finally:
        # Restore stdout and close log file
        sys.stdout = tee.terminal
        tee.close()
        print(f"\n✓ Test log saved to: {log_file.absolute()}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
