#!/usr/bin/env python3
"""
Check CUDA availability and GPU information.
"""

import torch

print("=== CUDA Information ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
else:
    print("CUDA is not available. Possible reasons:")
    print("1. PyTorch was installed without CUDA support")
    print("2. NVIDIA drivers are not installed or outdated")
    print("3. CUDA toolkit is not installed")
    print("4. GPU is not CUDA-compatible")
    
    # Check if we have a CPU-only PyTorch installation
    print(f"\nCurrent PyTorch installation supports: {torch.version.cuda if torch.version.cuda else 'CPU only'}")
