#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_SIZE 16

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int K, int N) {
    // Shared memory tiles - this is the fast on-chip memory
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Collaboratively load one tile of A and one tile of B
        // into shared memory. Each thread loads exactly one element.
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Wait for ALL threads in the block to finish loading
        // before any thread starts computing. This is critical.
        __syncthreads();

        // Compute partial sum using the tile in shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Wait for all threads to finish computing before
        // loading the next tile. Without this, some threads
        // might overwrite the tile while others are still using it.
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

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

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_gpu = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);

    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    matmul_cpu(h_A, h_B, h_C_cpu, M, K, N);

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

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
    return 0;
}
