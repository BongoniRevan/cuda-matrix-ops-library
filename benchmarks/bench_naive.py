import subprocess
import numpy as np
import time

print("=== NumPy CPU Benchmark ===")
for size in [128, 256, 512, 1024]:
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)

    # Warmup
    _ = np.matmul(A, B)

    start = time.perf_counter()
    for _ in range(10):
        C = np.matmul(A, B)
    end = time.perf_counter()

    avg_ms = (end - start) / 10 * 1000
    print(f"Size {size}x{size}: {avg_ms:.2f} ms")
