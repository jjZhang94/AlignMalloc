/*
** 
** analyze the memory access pattern in the SyncMalloc
** the random walk runs after finishing dynamic allocation
** adjacency list is imported by the dynamic allocation
**
nvcc -rdc=true -arch=sm_80 randomAccessSyncMalloc.cu
*/

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define BLOCKSIZE  128
#define BLOCKNUMBER 1

#define NODEPADDING (THREADADJACENCY/ADJACENCYLISTNUMBER)  //Each node has an independence padding 
#define NODENUMBER (8192*2)  // The node information can be read from a fixed file or set by the user
#define THRESHOLD 0.5  // Clustering weight set
#define ADJACENCYLISTNUMBER 64 // Each node has the maximum number, this also can be read from dynamic allocation
#define RANDOMWALKSTEP 50
#define THREADADJACENCY (1024*64)

__device__  //waiting function  
void sleep(clock_t cycles)
{
  clock_t start = clock();
  clock_t now;

  for (;;) {
    now = clock();
    clock_t spent = now > start ? now - start : now + (0xffffffff - start);
    if (spent >= cycles) {
      break;
    }
  }
}

__global__ void randomWalk(char* adjacencyAddress, int* indexSelect)
{
    //   clock_t cycles = 1000;   //waiting cycles
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % warpSize; 
    int warpId = index / warpSize;
    
    char* startAddress = adjacencyAddress + warpId * (THREADADJACENCY*32);
    printf("index %d, laneID, %d Array: %d\n", index, laneId, indexSelect[laneId]);
    startAddress = startAddress + indexSelect[laneId] * THREADADJACENCY;
    *startAddress = *startAddress + 1;
    // for(int i = 0; i < RANDOMWALKSTEP; i++)
    // {
    //     int indexThr = indexSelect[i];
    //     char *tmpAddress = startAddress + indexThr * NODEPADDING;
    //     float *tmpAddressFloat = (float *)(tmpAddress + sizeof(int));
    //     int *tmpAddressAggre = (int *) (tmpAddress + sizeof(int) + sizeof(float));
    //     *tmpAddressAggre = *tmpAddressAggre + 1;
    // }
} 

int main(void)
{
    //timing funciton
    struct timeval start, end;
    
    
    long long N = 1024LL*1024LL*1024LL*1;
    char *adjacencyAddress;
   
    cudaMallocManaged(&adjacencyAddress, N*sizeof(char));
    memset(adjacencyAddress, 0, N*sizeof(char));

    int *d_indexSelect;
    // int indexSelect[RANDOMWALKSTEP];
    int indexSelect[32] = {0,1,2,4, 6, 8, 9, 10, 12, 16, 17, 18,20,26,29, 30,25,23,21,19,15,14,13,11,7,5,3,28,27,24,31,22};
    // int indexSelect[32] = {0,1,2,4, 8, 9, 10, 16, 31,30,29,28,27,26,25,24,23, 22,21,20,19,18,17,15,14,13,12,11,10, 9,7,6,5,3};
    // int indexSelect[32];
    // for(int i = 0; i< 32; i++)
    // {
    //   indexSelect[i] = i;
    // }

    // Allocate memory on the device
    cudaMalloc(&d_indexSelect, 32 * sizeof(int));
    //create in indexSelect a random order
    srand(time(NULL));  //init a random generator
    // indexSelect[0] = 0;
    // indexSelect[1] = 1;
    // indexSelect[2] = 2;
    // indexSelect[3] = 4;
    // indexSelect[4] = 8;
    // indexSelect[5] = 16;
    // for(int i = 0; i < RANDOMWALKSTEP; i++)
    // {
    //     indexSelect[i] = rand() % (ADJACENCYLISTNUMBER);
    // }
    // Copy the index array to the device
    cudaMemcpy(d_indexSelect, indexSelect, 32 * sizeof(int), cudaMemcpyHostToDevice);
    
    //random generate adjacency list
    // for(int i = 0; i < BLOCKNUMBER*BLOCKSIZE; i++)
    // {
    //     for(int j = 0; j < ADJACENCYLISTNUMBER; j++)
    //     {
    //         char *startAddress = adjacencyAddress + i * THREADADJACENCY + j * NODEPADDING;
    //         int *startAddressInt = (int *)startAddress;
    //         float *startAddressFloat = (float *) (startAddress + sizeof(int));
    //         //set node Or read from the memory from dynamic allocation
    //         *startAddressInt = (rand() % (1024));
    //         //set weight
    //         *startAddressFloat = (float) rand() / RAND_MAX;
            
    //     } 
    // }

    gettimeofday(&start, NULL);
    randomWalk<<<BLOCKNUMBER, BLOCKSIZE>>>(adjacencyAddress, d_indexSelect);

    cudaDeviceSynchronize();

    cudaFree(adjacencyAddress);

    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double elapsed = seconds * 1000.0 + microseconds / 1000.0;

    printf("Time spent: %.3f milliseconds\n", elapsed);
    return 0;
    }
