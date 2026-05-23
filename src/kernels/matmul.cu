#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Naive matrix multiplication kernel
// Each thread computes one element of the output matrix C
// C = A * B where A is MxK, B is KxN, C is MxN
__global__ void matmul_naive(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU reference implementation to verify correctness
void matmul_cpu(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int M = 256, K = 256, N = 256;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_gpu = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);

    // Initialize matrices with random values
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (M + 15) / 16);
    matmul_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    // Copy result back
    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    // Verify against CPU
    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);

    // Check correctness
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (error > max_error) max_error = error;
    }

    printf("Matrix size: %dx%d\n", M, N);
    printf("Max error vs CPU: %e\n", max_error);
    if (max_error < 1e-3) {
        printf("CORRECTNESS CHECK PASSED\n");
    } else {
        printf("CORRECTNESS CHECK FAILED\n");
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    return 0;
}
