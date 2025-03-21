#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>

#include <time.h>

#define THREAD_ALLOCATION_SIZE (1024*1)
#define ACCESS_TIMES 100

//command line nvcc -rdc=true -arch=sm_80 randomAccessAfterAllocOurMethod.cu  MMUOnCPU.cu -lpthread

/* -- test for version 1.2 --*/
__global__ void waitForHostAndContinue(struct MMUOnTransfer** pMMUOnTransfer, struct MetaDataAllocationGPU* d_MetaDataAllocationGPU)
{
    char* a = (char*)allocateThr(THREAD_ALLOCATION_SIZE, pMMUOnTransfer, d_MetaDataAllocationGPU);  
    // a = a + WARPGAP*1 + 28;
    // char* b = (char*)addressAccess(a, d_MetaDataAllocationGPU);
    // *b = *b + 1;
    // int gap = b - d_MetaDataAllocationGPU[(threadIdx.x + blockIdx.x * blockDim.x) / warpSize].startAddress;
    // printf("Athread: %d, blockgap: %d, offset: %d\n", threadIdx.x + blockIdx.x * blockDim.x, gap/WARPGAP, gap%WARPGAP);
}

__global__ void accessMemory(struct MetaDataAllocationGPU* d_MetaDataAllocationGPU, int* d_randomAccessOrder)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = idx / warpSize;

    char* a = (char*)(d_MetaDataAllocationGPU[warpId].startAddress);
    
    for(int i = 0; i < ACCESS_TIMES; i++)
    {
        char* tmpPointer = (char *)(addressAccess(a + d_randomAccessOrder[i], d_MetaDataAllocationGPU));
        *tmpPointer = *tmpPointer + 1;
    }
}

int main() 
{  
    /*random access order*/
    int randomAccessOrder[ACCESS_TIMES];
    int* d_randomAccessOrder;
    srand(time(NULL));
    int blockNumber = ceil((float)THREAD_ALLOCATION_SIZE/ WARPGAP);
    for(int i = 0; i < ACCESS_TIMES; i++)
    {
        int randomBlock;
        if(blockNumber == 1)
        {
            randomBlock = 0;
        }else
        {
            randomBlock = rand() % (blockNumber);
        }
        
        int randomOffset = rand() % (WARPGAP - 100);
        // int randomBlock = rand() % (
        randomAccessOrder[i] = randomBlock * WARPGAP + randomOffset;
    }
    cudaMalloc(&d_randomAccessOrder, sizeof(int) * ACCESS_TIMES);
    cudaMemcpy(d_randomAccessOrder, randomAccessOrder, sizeof(int) * ACCESS_TIMES, cudaMemcpyHostToDevice);
    /* -- init meta -- */
    //init metadata in devices
    MetaDataAllocationGPU* d_MetaDataAllocationGPU;
    cudaMalloc(&d_MetaDataAllocationGPU, WARPNUMBER * sizeof(struct MetaDataAllocationGPU));
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

    cudaEvent_t start, stop, startAccess, stopAccess;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&startAccess);
    cudaEventCreate(&stopAccess);

    // Record the start event
    cudaEventRecord(start, 0);
    // Launch the kernel
    waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE, 0, stream>>>(d_pMMUOnTransfer, d_MetaDataAllocationGPU);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop); 
    aLaunchKernel(&args, stream);

    // Record the start event
    cudaEventRecord(startAccess);
    accessMemory<<<BLOCKNUMBER, BLOCKSIZE, 0>>>(d_MetaDataAllocationGPU, d_randomAccessOrder);
    cudaEventRecord(stopAccess);
    cudaEventSynchronize(stopAccess);
    // Record the stop event
    // Wait for the kernel to complete

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startAccess, stopAccess);
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