#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void relu_kernel(float* input, float* output, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

void relu_cpu(float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

int main() {
    int n = 1024 * 1024; // 1M elements
    size_t size = n * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);

    // Fill with values between -1 and 1
    for (int i = 0; i < n; i++)
        h_input[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    relu_cpu(h_input, h_output_cpu, n);

    float max_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float error = fabs(h_output_gpu[i] - h_output_cpu[i]);
        if (error > max_error) max_error = error;
    }

    printf("ReLU on %d elements\n", n);
    printf("Max error vs CPU: %e\n", max_error);
    printf("Average GPU time: %.4f ms\n", ms / 100);
    if (max_error < 1e-6)
        printf("CORRECTNESS CHECK PASSED\n");
    else
        printf("CORRECTNESS CHECK FAILED\n");

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output_gpu); free(h_output_cpu);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
