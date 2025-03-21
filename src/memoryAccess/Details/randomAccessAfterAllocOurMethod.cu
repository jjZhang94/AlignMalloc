#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>

#include <time.h>

#define THREAD_ALLOCATION_SIZE (1024*1024)
#define ACCESS_TIMES 100

#define NODEPADDING (THREADADJACENCY/ADJACENCYLISTNUMBER)  //Each node has an independence padding 
#define NODENUMBER (8192*2)  // The node information can be read from a fixed file or set by the user
#define THRESHOLD 0.5  // Clustering weight set
#define ADJACENCYLISTNUMBER 16 // Each node has the maximum number, this also can be read from dynamic allocation
#define RANDOMWALKSTEP 50
#define THREADADJACENCY (1024*1024)

// nvcc -rdc=true randomAccessAfterAllocOurMethod.cu  MMUOnCPU.cu -lpthread

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

__global__ void randomWalk(char* adjacencyAddress, int* indexSelect, char** d_warpStartAddress, int* indexLane)
{
    //   clock_t cycles = 1000;   //waiting cycles
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = index / warpSize;

    char* startAddress = d_warpStartAddress[warpId];
    for(int i = 0; i < RANDOMWALKSTEP; i++)
    {
        // int indexThr = indexSelect[i];
        int indexThr = indexSelect[i];
        char *tmpAddress = startAddress + indexThr * NODEPADDING;

        //addree Translation
        int laneId = threadIdx.x % warpSize; 
        laneId = indexLane[laneId];

        unsigned int offsetAddress = indexThr * NODEPADDING;
        int offsetBlock = offsetAddress/WARPGAP;
        char* returnPointer = tmpAddress + (offsetBlock * warpSize + laneId - offsetBlock) * WARPGAP;

        float *tmpAddressFloat = (float *)(returnPointer + sizeof(int));
        int *tmpAddressAggre = (int *) (returnPointer + sizeof(int) + sizeof(float));
        *tmpAddressAggre = *tmpAddressAggre + 1;
    }
} 

int main() 
{      
    long long N = 1024LL*1024LL*1024LL*32LL;
    char *adjacencyAddress;
   
    cudaMallocManaged(&adjacencyAddress, N*sizeof(char));
    memset(adjacencyAddress, 0, N*sizeof(char));

    int *d_indexSelect;
    int indexSelect[RANDOMWALKSTEP];

    char** d_warpStartAddress;
    char* warpStartAddress[WARPNUMBER];
    warpStartAddress[0] = adjacencyAddress;
    int gap = 0;
    if((THREADADJACENCY * WARPSIZE) < VABLOCK_SIZE)
    {
        gap = VABLOCK_SIZE;
    }else
    {
        int dd = (int)ceil((float)(THREADADJACENCY * WARPSIZE) / VABLOCK_SIZE);
        gap = dd * VABLOCK_SIZE;
    }
    for(int i = 1; i < WARPNUMBER; i++)
    {
        warpStartAddress[i] = warpStartAddress[i-1] + gap;
    }
    cudaMalloc(&d_warpStartAddress, WARPNUMBER * sizeof(char*));
    cudaMemcpy(d_warpStartAddress, warpStartAddress, WARPNUMBER * sizeof(char*), cudaMemcpyHostToDevice);

    // Allocate memory on the device
    cudaMalloc(&d_indexSelect, RANDOMWALKSTEP * sizeof(int));
    //create in indexSelect a random order
    srand(time(NULL));  //init a random generator
    // indexSelect[0] = 1;
    // indexSelect[1] = 1;
    // indexSelect[2] = 2;
    // indexSelect[3] = 4;
    // indexSelect[4] = 8;
    // indexSelect[5] = 16;
    int indexLane[32] = {0,1,2,4, 6, 8, 9, 10, 12, 16, 17, 18,20,26,29, 30,25,23,21,19,15,14,13,11,7,5,3,28,27,24,31,22};
    int* d_indexLane;
    cudaMalloc(&d_indexLane, 32 * sizeof(int));
    cudaMemcpy(d_indexLane, indexLane, 32 * sizeof(int), cudaMemcpyHostToDevice);

    for(int i = 0; i < RANDOMWALKSTEP; i++)
    {
        indexSelect[i] = rand() % (ADJACENCYLISTNUMBER);
    }
    // Copy the index array to the device
    cudaMemcpy(d_indexSelect, indexSelect, RANDOMWALKSTEP * sizeof(int), cudaMemcpyHostToDevice);
    
    //random generate adjacency list
    for(int i = 0; i < BLOCKNUMBER*BLOCKSIZE; i++)
    {
        for(int j = 0; j < ADJACENCYLISTNUMBER; j++)
        {
            char *startAddress = adjacencyAddress + i * THREADADJACENCY + j * NODEPADDING;
            int *startAddressInt = (int *)startAddress;
            float *startAddressFloat = (float *) (startAddress + sizeof(int));
            //set node Or read from the memory from dynamic allocation
            *startAddressInt = (rand() % (1024));
            //set weight
            *startAddressFloat = (float) rand() / RAND_MAX;
            
        } 
    }
    
    randomWalk<<<BLOCKNUMBER, BLOCKSIZE>>>(adjacencyAddress, d_indexSelect, d_warpStartAddress, d_indexLane);

    cudaDeviceSynchronize();

    cudaFree(adjacencyAddress);

    return 0;
}