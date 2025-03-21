#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>

#include <time.h>

//command line nvcc -rdc=true -arch=sm_80 singleAllocation.cu  MMUOnCPU.cu -lpthread

/* -- test for version 1.2 --*/
__global__ void waitForHostAndContinue(struct MMUOnTransfer** pMMUOnTransfer)
{
    int* a = (int*)allocateThr(256*1024, pMMUOnTransfer);    
}

int main() 
{/* -- init meta -- */
    // int *d_isWarpBranch;
    // cudaMalloc(&d_isWarpBranch, WARPNUMBER * sizeof(int));
    // init MemoryManagement
    MemoryManagement* memoryManagement[ALLOCATIONMANAGEMENTTHREADNUMBER];
    struct MMUOnTransfer** pMMUOnTransfer;
    struct MMUOnTransfer** d_pMMUOnTransfer;
    pMMUOnTransfer =  (MMUOnTransfer **)malloc(WARPNUMBER * sizeof(MMUOnTransfer *));
    pthread_t thread_id;
    thread_args args;
    int should_exit = 0;
    for(int i = 0; i < WARPNUMBER; i++)
    {
        cudaError_t cudaStatus = cudaMallocManaged(&(pMMUOnTransfer[i]), sizeof(struct MMUOnTransfer));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
            return 1;
        }
    }

    //init MMUOnTransfer  
    initAllocationStru(memoryManagement, pMMUOnTransfer, &thread_id, &args, &should_exit);

    cudaMalloc(&d_pMMUOnTransfer, WARPNUMBER * sizeof(MMUOnTransfer *));
    cudaMemcpy(d_pMMUOnTransfer, pMMUOnTransfer, WARPNUMBER * sizeof(MMUOnTransfer *), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    // Launch the kernel
    waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE, 0, stream>>>(d_pMMUOnTransfer);

    // printf("SSSSS\n");
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop); 
    aLaunchKernel(&args, stream);
    // printf("SSSSsssssS\n");
    // Record the stop event
    // Wait for the kernel to complete

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time:  %f  milliseconds\n", milliseconds);
    for(int i = 0; i<3; i++)
    {
        printf("pMMUOnTransfer %d\n", (pMMUOnTransfer[i])->sizeAllocate);
    }

    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 1*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress + 2*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 3*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 4*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 5*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 6*PAGE_SIZE));
    // printf("%d\n", *(int*)((char*)pMMUOnTransfer -> bitmapStartAddress+ 7*PAGE_SIZE));
    
    for(int i = 0; i<WARPNUMBER; i++)
    {
        cudaFree(pMMUOnTransfer[i]);
    }

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}