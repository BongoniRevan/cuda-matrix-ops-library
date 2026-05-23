import ctypes
import numpy as np
import os

lib_path = os.path.join(os.path.dirname(__file__), '..', 'libmatops.so')
lib = ctypes.CDLL(lib_path)

lib.cuda_matmul.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.cuda_matmul.restype = None

lib.cuda_relu.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int
]
lib.cuda_relu.restype = None

lib.cuda_softmax.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int
]
lib.cuda_softmax.restype = None

def matmul(A, B):
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible matrix dimensions"
    C = np.zeros((M, N), dtype=np.float32)
    lib.cuda_matmul(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, K, N
    )
    return C

def relu(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros_like(x)
    lib.cuda_relu(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.size
    )
    return out

def softmax(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, cols = x.shape
    out = np.zeros_like(x)
    lib.cuda_softmax(
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rows, cols
    )
    return out
