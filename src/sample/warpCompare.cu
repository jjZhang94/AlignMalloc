#include <cuda_runtime.h>
#include <stdio.h>

//nvcc -arch=sm_80 warpCompare.cu
// Kernel function to check if all threads in a warp have the same value
__global__ void checkUniformity(int* input, bool* output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = idx / warpSize;  // Warp index
    int laneId = threadIdx.x % warpSize;  // Lane index within the warp

    int value = input[idx];  // Load the input value for this thread

    // Use __match_all_sync to check if all active threads have the same value
    unsigned int mask = 0xffffffff;  // Full warp participation
    int pred = 0;
    unsigned int match = __match_all_sync(mask, value, &pred);

    // The __match_all_sync function sets the mask to all bits 1 if all match
    //bool allUniform = (mask == 0xffffffff);

    // Only the first thread in each warp writes the result
    if (laneId == 0) {
        output[warpId] = pred;
    }
}

int main() {
    int warpSize = 32;
    int n = 64; // Total number of threads (must be multiple of 32 for this example)
    int numWarps = n / warpSize;

    int* h_input = new int[n];
    bool* h_output = new bool[numWarps];

    // Initialize input data
    for (int i = 0; i < n; ++i) {
        h_input[i] = i / warpSize;  // Same value within each warp, change to test different values
    }

    // h_input[1] = 2;
    int *d_input;
    bool *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, numWarps * sizeof(bool));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    checkUniformity<<<1, n>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, numWarps * sizeof(bool), cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < numWarps; ++i) {
        printf("Warp %d uniformity: %s\n", i, h_output[i] ? "true" : "false");
    }

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_output;

    return 0;
}