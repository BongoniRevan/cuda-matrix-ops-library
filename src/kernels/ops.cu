#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        tileA[threadIdx.y][threadIdx.x] = (row < M && t * TILE_SIZE + threadIdx.x < K)
            ? A[row * K + t * TILE_SIZE + threadIdx.x] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (col < N && t * TILE_SIZE + threadIdx.y < K)
            ? B[(t * TILE_SIZE + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

__global__ void relu_kernel(float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = fmaxf(0.0f, input[i]);
}

__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row < rows) {
        float* in = input + row * cols;
        float* out = output + row * cols;
        float max_val = in[0];
        for (int j = 1; j < cols; j++)
            if (in[j] > max_val) max_val = in[j];
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            out[j] = expf(in[j] - max_val);
            sum += out[j];
        }
        for (int j = 0; j < cols; j++) out[j] /= sum;
    }
}

extern "C" {
    void cuda_matmul(float* A, float* B, float* C, int M, int K, int N) {
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_tiled_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
        cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }

    void cuda_relu(float* input, float* output, int n) {
        float *d_in, *d_out;
        cudaMalloc(&d_in, n * sizeof(float));
        cudaMalloc(&d_out, n * sizeof(float));
        cudaMemcpy(d_in, input, n * sizeof(float), cudaMemcpyHostToDevice);
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        relu_kernel<<<blocks, threads>>>(d_in, d_out, n);
        cudaMemcpy(output, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out);
    }

    void cuda_softmax(float* input, float* output, int rows, int cols) {
        float *d_in, *d_out;
        size_t size = rows * cols * sizeof(float);
        cudaMalloc(&d_in, size);
        cudaMalloc(&d_out, size);
        cudaMemcpy(d_in, input, size, cudaMemcpyHostToDevice);
        softmax_kernel<<<rows, 1>>>(d_in, d_out, rows, cols);
        cudaMemcpy(output, d_out, size, cudaMemcpyDeviceToHost);
        cudaFree(d_in); cudaFree(d_out);
    }
}
