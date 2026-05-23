# CUDA Matrix Operations Library

A high-performance matrix operations library written from scratch in CUDA C++, exposing GPU-accelerated operations to Python via ctypes. Implements matrix multiplication, ReLU, and Softmax without relying on cuBLAS or any external CUDA math libraries.

Built as a portfolio project to demonstrate GPU programming fundamentals at the level relevant to systems and ML infrastructure engineering roles.

---

## Operations

- Matrix Multiplication: naive kernel to shared memory tiled kernel
- ReLU activation: element-wise parallel kernel
- Softmax: numerically stable row-wise kernel

---

## Key CUDA Concepts Demonstrated

- Thread hierarchy: grids, blocks, warps, and threads
- Shared memory tiling for cache-efficient matrix multiplication
- `__syncthreads()` for intra-block synchronization
- Host and device memory management with `cudaMalloc` and `cudaMemcpy`
- CUDA event-based kernel timing
- Numerical stability in GPU floating point operations
- Python interop via ctypes shared library

---

## Performance Results (NVIDIA Tesla T4, CUDA 12)

### Matrix Multiplication: CUDA Tiled Kernel vs NumPy CPU

| Matrix Size | NumPy (ms) | CUDA Tiled (ms) | Speedup |
|-------------|------------|-----------------|---------|
| 128x128     | 0.06       | 0.44            | 0.14x   |
| 256x256     | 0.83       | 0.85            | 0.97x   |
| 512x512     | 9.22       | 5.60            | 1.65x   |
| 1024x1024   | 39.82      | 9.82            | 4.05x   |
| 2048x2048   | 226.15     | 44.39           | 5.10x   |

CUDA shows no advantage at small matrix sizes due to kernel launch overhead and memory transfer costs. The GPU advantage becomes significant at 512x512 and grows strongly with matrix size, reaching 5.10x at 2048x2048. This is expected behavior and demonstrates why GPUs are used for large-scale matrix operations in deep learning.

---

### Naive vs Shared Memory Tiled Kernel (GPU only, CUDA events)

| Matrix Size | Naive (ms) | Tiled (ms) | Speedup |
|-------------|------------|------------|---------|
| 256x256     | 0.143      | 0.097      | 1.47x   |
| 512x512     | 1.042      | 0.675      | 1.54x   |
| 1024x1024   | 8.336      | 5.258      | 1.59x   |
| 2048x2048   | 34.252     | 20.342     | 1.68x   |

Shared memory tiling consistently outperforms the naive kernel across all sizes. The speedup grows with matrix size as larger matrices benefit more from reduced global memory traffic. The tiled kernel reuses each loaded value TILE_SIZE times from fast on-chip shared memory instead of reading from slow global memory on every access.

---

### ReLU and Softmax

| Operation | Input Size  | NumPy (ms) | CUDA (ms) | Note                   |
|-----------|-------------|------------|-----------|------------------------|
| ReLU      | 1M elements | 0.88       | 2.88      | PCIe transfer overhead |
| Softmax   | 1024 x 512  | 2.02       | 2.04      | Near parity            |

ReLU and Softmax show near parity or overhead at these sizes due to PCIe memory transfer cost between CPU and GPU. In a real inference pipeline, data stays resident on the GPU across operations and these transfers are eliminated, which is where GPU pipelines achieve their true throughput advantage.

---

## Project Structure

    cuda-matops/
    ├── src/
    │   └── kernels/
    │       ├── matmul.cu               Naive matrix multiplication kernel
    │       ├── matmul_tiled.cu         Shared memory tiled matmul kernel
    │       ├── relu.cu                 ReLU activation kernel
    │       ├── softmax.cu              Numerically stable softmax kernel
    │       └── ops.cu                  Combined ops compiled as shared library
    ├── python/
    │   └── matops.py                   Python bindings via ctypes
    ├── tests/
    │   └── test_all_ops.py             Correctness verification vs NumPy
    ├── benchmarks/
    │   ├── bench_naive.py              NumPy CPU baseline
    │   ├── bench_cuda_timing.cu        CUDA event benchmarks
    │   └── final_benchmark.py          Full CUDA vs NumPy comparison
    └── notes/
        └── roofline_analysis.md        Performance analysis writeup

---

## Environment

- GPU: NVIDIA Tesla T4
- CUDA: 12.2
- Python: 3.10
- OS: Linux (Google Colab)

---

## How to Run

Compile shared library:

    nvcc src/kernels/ops.cu -o libmatops.so --shared -Xcompiler -fPIC -arch=sm_75

Compile CUDA timing benchmark:

    nvcc benchmarks/bench_cuda_timing.cu -o bench_timing -arch=sm_75

Run correctness tests:

    python tests/test_all_ops.py

Run full benchmark:

    python benchmarks/final_benchmark.py

---

## Key Engineering Insight

The benchmark results tell an important story about GPU computing that goes beyond raw speedup numbers.

Small matrices are CPU-friendly because the overhead of launching a CUDA kernel and transferring data over PCIe dominates the actual computation time. Large matrices are where GPUs dominate because the parallel computation time dwarfs the fixed overhead costs.

This is precisely why frameworks like PyTorch batch operations and keep tensors resident on the GPU across an entire forward pass rather than transferring data back and forth per operation.

---

## What I Learned

Writing CUDA kernels from scratch made the GPU memory hierarchy tangible in a way that using PyTorch never does. The single biggest insight was understanding why the naive matmul kernel is slow: not because the GPU is doing wrong math, but because 65,536 threads are all hammering global memory independently when they could be sharing data through shared memory.

The roofline analysis in notes/roofline_analysis.md shows that matrix multiplication at large sizes is theoretically compute-bound on the T4, but the naive kernel behaves as memory-bound because of redundant global memory traffic. Shared memory tiling closes that gap by reducing effective memory traffic by a factor of TILE_SIZE.
