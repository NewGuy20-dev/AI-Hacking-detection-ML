#!/usr/bin/env python3
"""Quick test of adversarial stress testing enhancements."""
import sys
from pathlib import Path

# Add stress test v14 directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from difficulty import DifficultyMixin
from real_data import RealDataLoader
from scenarios import PayloadGenerator, URLGenerator, TimeSeriesGenerator, BenignAdversarialGenerator

def test_difficulty_mixin():
    """Test DifficultyMixin obfuscation."""
    print("=" * 60)
    print("Testing DifficultyMixin")
    print("=" * 60)
    
    mixin = DifficultyMixin()
    payload = "' OR '1'='1"
    
    print(f"\nOriginal payload: {payload}")
    for difficulty in ['easy', 'medium', 'hard', 'adversarial']:
        obfuscated = mixin.apply_difficulty(payload, difficulty, 'payload')
        print(f"{difficulty.capitalize():12s}: {obfuscated}")

def test_real_data_loader():
    """Test RealDataLoader."""
    print("\n" + "=" * 60)
    print("Testing RealDataLoader")
    print("=" * 60)
    
    loader = RealDataLoader()
    
    print(f"\nAvailable categories: {loader.get_available_categories()}")
    
    print("\nSample SQLi payloads:")
    sqli_samples = loader.sample('sqli', 3)
    for i, sample in enumerate(sqli_samples, 1):
        print(f"  {i}. {sample[:80]}")
    
    print("\nSample phishing URLs:")
    phishing_samples = loader.sample('phishing_url', 3)
    for i, sample in enumerate(phishing_samples, 1):
        print(f"  {i}. {sample[:80]}")

def test_generators():
    """Test enhanced generators."""
    print("\n" + "=" * 60)
    print("Testing Enhanced Generators")
    print("=" * 60)
    
    # Test PayloadGenerator
    print("\nPayloadGenerator (5 samples):")
    payload_gen = PayloadGenerator(seed=42)
    scenarios = payload_gen.generate(5, {'sqli': 0.5, 'xss': 0.5})
    for s in scenarios:
        print(f"  {s.difficulty:12s} | {s.category:8s} | {s.input_data[:60]}")
    
    # Test URLGenerator
    print("\nURLGenerator (5 samples):")
    url_gen = URLGenerator(seed=42)
    scenarios = url_gen.generate(5, {'phishing': 0.5, 'typosquatting': 0.5})
    for s in scenarios:
        print(f"  {s.difficulty:12s} | {s.category:15s} | {s.input_data[:60]}")
    
    # Test TimeSeriesGenerator
    print("\nTimeSeriesGenerator (4 samples - one per difficulty):")
    ts_gen = TimeSeriesGenerator(seed=42)
    scenarios = ts_gen.generate(4, {'ddos': 1.0})
    for s in scenarios:
        print(f"  {s.difficulty:12s} | {s.category:8s} | shape: {s.input_data.shape}")
    
    # Test BenignAdversarialGenerator
    print("\nBenignAdversarialGenerator (5 samples):")
    benign_gen = BenignAdversarialGenerator(seed=42)
    scenarios = benign_gen.generate(5)
    for s in scenarios:
        print(f"  {s.category:20s} | expected={s.expected_label} | {s.input_data[:60]}")

if __name__ == '__main__':
    print("\n🔬 Adversarial Stress Testing - Component Tests\n")
    
    try:
        test_difficulty_mixin()
        test_real_data_loader()
        test_generators()
        
        print("\n" + "=" * 60)
        print("✅ All component tests passed!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run full stress test: python scripts/stress_test_v14.py --model payload")
        print("  2. Check accuracy drops from 100% to 80-90%")
        print("  3. Verify per-difficulty breakdown in output")
        print("  4. View dashboard for visual confirmation")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
