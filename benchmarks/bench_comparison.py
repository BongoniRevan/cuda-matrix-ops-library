import subprocess
import numpy as np
import time
import ctypes
import os

print("=== Timing Comparison: Naive vs Tiled vs NumPy ===\n")

print("NumPy CPU baseline:")
for size in [256, 512, 1024, 2048]:
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    _ = np.matmul(A, B)
    start = time.perf_counter()
    for _ in range(5):
        C = np.matmul(A, B)
    end = time.perf_counter()
    avg_ms = (end - start) / 5 * 1000
    print(f"  {size}x{size}: {avg_ms:.2f} ms")
