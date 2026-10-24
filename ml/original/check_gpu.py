#!/usr/bin/env python3
"""check_gpu.py - 10-second check that PyTorch can see and use the GPU."""
import sys
import time

try:
    import torch
except ImportError:
    sys.exit("PyTorch is not installed. Use a RunPod PyTorch template, or run: pip install torch")

print(f"PyTorch {torch.__version__}, CUDA build {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("No GPU visible to PyTorch. On RunPod: check the pod has a GPU and you used a PyTorch/CUDA template.")

name = torch.cuda.get_device_name(0)
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name} ({mem:.0f} GB)")

x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    y = x @ x
torch.cuda.synchronize()
secs = time.time() - t0
print(f"Matrix multiply test: {20 * 2 * 4096 ** 3 / secs / 1e12:.1f} TFLOPS - GPU is working.")
