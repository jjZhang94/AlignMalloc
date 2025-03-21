/*
** 
** analyze the memory access pattern in the SyncMalloc
** the random walk runs after finishing dynamic allocation
** adjacency list is imported by the dynamic allocation
**
*/

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#define NODEPADDING 4096  //Each node has an independence padding 
#define NODENUMBER 8192*2  // The node information can be read from a fixed file or set by the user
#define THRESHOLD 0.5  // Clustering weight set
#define ADJACENCYLISTNUMBER 512 // Each node has the maximum number, this also can be read from dynamic allocation
#define BLOCKSIZE  64
#define BLOCKNUMBER 1
#define RANDOMWALKSTEP 6
#define THREADADJACENCY NODEPADDING*ADJACENCYLISTNUMBER

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

// __global__ void randomWalk(char* adjacencyAddress, int* indexSelect)
// {
//     //   clock_t cycles = 1000;   //waiting cycles
//     int index = blockIdx.x * blockDim.x + threadIdx.x;

//     char* startAddress = adjacencyAddress + index * THREADADJACENCY;
//     for(int i = 0; i < RANDOMWALKSTEP; i++)
//     {
//         int indexThr = indexSelect[i];
//         char *tmpAddress = startAddress + indexThr * NODEPADDING;
//         float *tmpAddressFloat = (float *)(tmpAddress + sizeof(int));
//         int *tmpAddressAggre = (int *) (tmpAddress + sizeof(int) + sizeof(float));
//         if(*tmpAddressFloat > THRESHOLD)
//         {
//             // printf("floate = %f\n", *tmpAddressFloat);
//             *tmpAddressAggre = 1;
//         }else
//         {
//             *tmpAddressAggre = 0;
//         }
//     }
// } 

__global__ void randomWalk(char* adjacencyAddress, int* indexSelect)
{
    //   clock_t cycles = 1000;   //waiting cycles
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    
    char* startAddress = adjacencyAddress + index * THREADADJACENCY;
    for(int i = 0; i < RANDOMWALKSTEP; i++)
    {
        int indexThr = indexSelect[i];
        char *tmpAddress = startAddress + indexThr * NODEPADDING;
        float *tmpAddressFloat = (float *)(tmpAddress + sizeof(int));
        int *tmpAddressAggre = (int *) (tmpAddress + sizeof(int) + sizeof(float));
        if(*tmpAddressFloat > THRESHOLD)
        {
            // printf("floate = %f\n", *tmpAddressFloat);
            *tmpAddressAggre = 1;
        }else
        {
            *tmpAddressAggre = 0;
        }
    }
} 


int main(void)
{
    //timing funciton
    struct timeval start, end;
    
    
    long long N = 1024*1024*1024;
    char *adjacencyAddress;
   
    cudaMallocManaged(&adjacencyAddress, N*sizeof(char));

    int *d_indexSelect;
    int indexSelect[RANDOMWALKSTEP];


    // Allocate memory on the device
    cudaMalloc(&d_indexSelect, RANDOMWALKSTEP * sizeof(int));
    //create in indexSelect a random order
    srand(time(NULL));  //init a random generator
    // for(int i = 0; i < RANDOMWALKSTEP; i++)
    // {
    //     indexSelect[i] = rand() % (ADJACENCYLISTNUMBER);
    // }
    indexSelect[0] = 0;
    indexSelect[1] = 1*16;
    indexSelect[2] = 2*16;
    indexSelect[3] = 4*16;
    indexSelect[4] = 8*16;
    indexSelect[5] = 16*16;
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
