#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define numElements 512

// command line nvcc -rdc=true -arch=sm_80 memcpy.cu
__global__ void testMemcpyKernel(int* dest, int* src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        memcpy(dest+idx, src+idx, 4);
    }
}

int main() {
    size_t size = numElements * sizeof(int);

    int h_src[512];
    int h_dest[512];

    // init
    for (int i = 0; i < numElements; ++i) {
        h_src[i] = i;
    }

    int* d_src;
    int* d_dest;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dest, size);

    cudaMemcpy(d_src, h_src, size, cudaMemcpyHostToDevice);

    testMemcpyKernel<<<1, 512>>>(d_dest, d_src);
    
    // 等待 GPU 完成所有任务
    cudaError_t error = cudaDeviceSynchronize();
    
    if (error != cudaSuccess) {
        // 检查是否有错误发生
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    } else {
        printf("GPU tasks have completed successfully\n");
    }

    cudaMemcpy(h_dest, d_dest, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < numElements; ++i) {
        if (h_dest[i] != h_src[i]) {
            printf("Mismatch at  %d was: %d  should be: %d\n", i, h_dest[i], h_src[i]);
            break;
        }
    }

    cudaFree(d_src);
    cudaFree(d_dest);

    return 0;
}

