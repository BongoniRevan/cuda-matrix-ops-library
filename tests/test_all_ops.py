import sys
sys.path.insert(0, '.')
import numpy as np
from python.matops import matmul, relu, softmax

print("=== Testing all operations from Python ===\n")

# Test matmul
A = np.random.rand(256, 256).astype(np.float32)
B = np.random.rand(256, 256).astype(np.float32)
C_cuda = matmul(A, B)
C_numpy = np.matmul(A, B)
err = np.max(np.abs(C_cuda - C_numpy))
print(f"MatMul  | Max error vs NumPy: {err:.6e} | {'PASS' if err < 1e-3 else 'FAIL'}")

# Test relu
x = np.random.randn(1000).astype(np.float32)
r_cuda = relu(x)
r_numpy = np.maximum(0, x)
err = np.max(np.abs(r_cuda - r_numpy))
print(f"ReLU    | Max error vs NumPy: {err:.6e} | {'PASS' if err < 1e-6 else 'FAIL'}")

# Test softmax
x = np.random.randn(128, 512).astype(np.float32)
s_cuda = softmax(x)
row_sums = s_cuda.sum(axis=1)
err = np.max(np.abs(row_sums - 1.0))
print(f"Softmax | Max row sum error:  {err:.6e} | {'PASS' if err < 1e-5 else 'FAIL'}")

print("\nAll operations callable from Python successfully.")
