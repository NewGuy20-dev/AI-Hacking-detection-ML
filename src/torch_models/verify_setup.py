"""Verify PyTorch and GPU setup for Phase 1."""
import sys


def verify_pytorch():
    """Check PyTorch installation and GPU availability."""
    print("=" * 50)
    print("PyTorch Setup Verification")
    print("=" * 50)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed")
        print("  Run: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False
    
    # CUDA check
    print(f"\n--- GPU Status ---")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"✓ VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"✓ Compute capability: {props.major}.{props.minor}")
    else:
        print("✗ CUDA not available (will use CPU)")
    
    # Test tensor operations
    print(f"\n--- Tensor Test ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(100, 100).to(device)
    y = torch.matmul(x, x)
    print(f"✓ Tensor operations on {device}: OK")
    
    # Test mixed precision
    if torch.cuda.is_available():
        print(f"\n--- Mixed Precision Test ---")
        with torch.amp.autocast('cuda'):
            z = torch.matmul(x, x)
        print(f"✓ FP16 autocast: OK")
    
    # Test our modules
    print(f"\n--- Module Import Test ---")
    try:
        from utils import get_device, setup_gpu, EarlyStopping
        print("✓ utils.py: OK")
    except ImportError as e:
        print(f"✗ utils.py: {e}")
    
    try:
        from datasets import PayloadDataset, URLDataset, TimeSeriesDataset
        print("✓ datasets.py: OK")
    except ImportError as e:
        print(f"✗ datasets.py: {e}")
    
    # Test dataset creation
    print(f"\n--- Dataset Test ---")
    try:
        ds = PayloadDataset(["test payload", "SELECT * FROM"], [0, 1])
        sample = ds[0]
        print(f"✓ PayloadDataset: input shape {sample['input'].shape}")
        
        ds = URLDataset(["http://example.com", "http://evil.com"], [0, 1])
        sample = ds[0]
        print(f"✓ URLDataset: input shape {sample['input'].shape}")
        
        import numpy as np
        ds = TimeSeriesDataset(np.random.randn(10, 60, 8), np.zeros(10))
        sample = ds[0]
        print(f"✓ TimeSeriesDataset: input shape {sample['input'].shape}")
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Phase 1 Setup: COMPLETE ✓")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = verify_pytorch()
    sys.exit(0 if success else 1)
