"""Report Generator - Aggregate results and generate reports."""
import json
from pathlib import Path
from datetime import datetime


def generate_report(
    training_results: dict,
    sanity_results: dict,
    holdout_results: dict = None,
    stress_results: dict = None,
    fp_results: dict = None,
    output_dir: Path = None
) -> dict:
    """Generate unified validation report."""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'training': training_results,
        'validation': {
            'sanity_check': sanity_results,
        },
        'deployment_ready': True,
        'warnings': []
    }
    
    # Add holdout results
    if holdout_results:
        report['validation']['holdout_evaluation'] = holdout_results
        if not holdout_results.get('passed', True):
            report['warnings'].append('Holdout evaluation below baseline')
    
    # Add stress test results
    if stress_results:
        report['validation']['stress_test'] = stress_results
    
    # Add FP test results
    if fp_results:
        report['validation']['fp_test'] = fp_results
        if not fp_results.get('passed', True):
            report['warnings'].append('FP rate above target')
    
    # Determine deployment readiness
    if not sanity_results.get('passed', False):
        report['deployment_ready'] = False
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\n  Sanity Check:      {'PASS' if sanity_results.get('passed') else 'FAIL'}")
    
    if holdout_results:
        print(f"  Holdout Eval:      {'PASS' if holdout_results.get('passed') else 'WARN'}")
    
    if stress_results:
        print(f"  Stress Test:       {'COMPLETE' if stress_results.get('ran') else 'SKIP'}")
    
    if fp_results:
        print(f"  FP Test:           {'PASS' if fp_results.get('passed') else 'WARN'}")
    
    print(f"\n  Deployment Ready:  {'YES' if report['deployment_ready'] else 'NO'}")
    
    if report['warnings']:
        print(f"\n  Warnings:")
        for w in report['warnings']:
            print(f"    - {w}")
    
    print("="*60)
    
    # Save report
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f'validation_report_{timestamp}.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n  Report saved: {report_file}")
    
    return report
