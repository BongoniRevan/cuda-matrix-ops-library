#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Each block handles one row of the matrix
// This is a simple row-wise softmax
__global__ void softmax_kernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;

    if (row < rows) {
        float* row_input = input + row * cols;
        float* row_output = output + row * cols;

        // Step 1: Find max value in this row for numerical stability
        float max_val = row_input[0];
        for (int j = 1; j < cols; j++) {
            if (row_input[j] > max_val)
                max_val = row_input[j];
        }

        // Step 2: Compute exponentials and their sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row_output[j] = expf(row_input[j] - max_val);
            sum += row_output[j];
        }

        // Step 3: Normalize
        for (int j = 0; j < cols; j++) {
            row_output[j] /= sum;
        }
    }
}

void softmax_cpu(float* input, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float* row_in = input + i * cols;
        float* row_out = output + i * cols;

        float max_val = row_in[0];
        for (int j = 1; j < cols; j++)
            if (row_in[j] > max_val) max_val = row_in[j];

        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row_out[j] = expf(row_in[j] - max_val);
            sum += row_out[j];
        }
        for (int j = 0; j < cols; j++)
            row_out[j] /= sum;
    }
}

int main() {
    int rows = 1024;
    int cols = 512;
    size_t size = rows * cols * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);

    for (int i = 0; i < rows * cols; i++)
        h_input[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // One block per row
    softmax_kernel<<<rows, 1>>>(d_input, d_output, rows, cols);

    cudaEventRecord(start);
    for (int i = 0; i < 100; i++)
        softmax_kernel<<<rows, 1>>>(d_input, d_output, rows, cols);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);
    softmax_cpu(h_input, h_output_cpu, rows, cols);

    float max_error = 0.0f;
    for (int i = 0; i < rows * cols; i++) {
        float error = fabs(h_output_gpu[i] - h_output_cpu[i]);
        if (error > max_error) max_error = error;
    }

    // Verify rows sum to 1.0
    float max_sum_error = 0.0f;
    for (int i = 0; i < rows; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < cols; j++)
            row_sum += h_output_gpu[i * cols + j];
        float sum_error = fabs(row_sum - 1.0f);
        if (sum_error > max_sum_error) max_sum_error = sum_error;
    }

    printf("Softmax on %dx%d matrix\n", rows, cols);
    printf("Max error vs CPU: %e\n", max_error);
    printf("Max row sum error (should be ~0): %e\n", max_sum_error);
    printf("Average GPU time: %.4f ms\n", ms / 100);
    if (max_error < 1e-5)
        printf("CORRECTNESS CHECK PASSED\n");
    else
        printf("CORRECTNESS CHECK FAILED\n");

    cudaFree(d_input); cudaFree(d_output);
    free(h_input); free(h_output_gpu); free(h_output_cpu);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
