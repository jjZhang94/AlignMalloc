#include <cuda_runtime.h>
#include <stdio.h>

/* 
Command line nvcc -arch=sm_80 warpSum.cu 
Attempting to use warpSize from host code will lead to errors. Only device functions or kernels can utilize warpSize.
*/
__global__ void sumWarpUsingReduceAdd(int *input, int *output, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % warpSize;

    // Assuming blockDim.x is a multiple of warpSize
    int warpId = index / warpSize;

    // Each thread loads one element from global to shared mem
    int value = (index < n) ? input[index] : 0;

    // Warp-wide reduction using __reduce_add_sync
    unsigned mask = 0xffffffff; // Full warp mask
    int sum = __reduce_add_sync(mask, value);

    // Write reduced value to global memory
    if (lane == 0) {
        output[warpId] = sum;
    }
}

int main() {
    int n = 64; // Example size
    int *d_input;
    int input[64];
    int *d_output;
    int output[2];


    cudaMalloc(&d_input, n * sizeof(int));
    for(int i = 0; i < n; i++)
    {
        input[i] = i;
    }
    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, (n / 32) * sizeof(int));

    // Assuming input is already filled
    sumWarpUsingReduceAdd<<<1, n>>>(d_input, d_output, n);

    cudaDeviceSynchronize();
    // Copy and print the output etc...
    cudaMemcpy(output, d_output, (n / 32) * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 2; i++)
    {
        printf("output: %d\n", output[i]);
    }
    
    cudaFree(input);
    cudaFree(output);

    return 0;
}