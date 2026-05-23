#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

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

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

void benchmark(int M, int K, int N) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float naive_ms = ms / 10;

    // Warmup
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float tiled_ms = ms / 10;

    printf("Size %4dx%4d | Naive: %8.3f ms | Tiled: %8.3f ms | Speedup: %.2fx\n",
           M, N, naive_ms, tiled_ms, naive_ms / tiled_ms);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    printf("%-20s | %-18s | %-18s | %s\n",
           "Matrix Size", "Naive (ms)", "Tiled (ms)", "Speedup");
    printf("%s\n", "---------------------------------------------------------------");
    benchmark(256,  256,  256);
    benchmark(512,  512,  512);
    benchmark(1024, 1024, 1024);
    benchmark(2048, 2048, 2048);
    return 0;
}
