#include <stdio.h>

#define ACCESS_NUMBER_GROUP 60
#define ACCESS_GROUP_SIZE 7
#define ACCESS_NUMBER (ACCESS_NUMBER_GROUP*ACCESS_GROUP_SIZE)
#define ACTUALACCESSTIME 60//the actual access time is no more than the total group number.
#define THREAD_LARGE_BLOCK (2*1024*1024)
#define THREAD_SMALL_BLOCK_SIZE (64*1024)
#define THREAD_SMALL_BLOCK_NUMBER 32
#define THREAD_GAP (2*1024*1024*ACCESS_NUMBER_GROUP)
//information for launching kernel 
#define BLOCK_SIZE 1024
#define BLOCK_NUMBER 1
#define THREADNUMBER (BLOCK_SIZE * BLOCK_NUMBER)
#define WAPRSIZE 32
#define WARP_NUMBER (32)

//command line nvcc -rdc=true -arch=sm_80 randomAccessOurOver.cu

__global__ void accessOver(int* d_innerGroupOrder, int* d_accessOrder, char** d_warpStartAddress, int* d_indexLane)
{
    //   clock_t cycles = 1000;   //waiting cycles
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = index / warpSize;
    int laneId = index % warpSize;
    laneId = d_indexLane[laneId];

    for(int i = 0; i < ACTUALACCESSTIME; i++)
    {
        char *tmpStartAddress = d_warpStartAddress[(ACCESS_NUMBER_GROUP * warpId + d_accessOrder[i])];
        
        for(int j = 0; j < ACCESS_GROUP_SIZE; j++)
        {
            //according to inner group order fetch
            char *tt = tmpStartAddress + d_innerGroupOrder[j] * THREAD_LARGE_BLOCK;
            tt = tt + laneId * THREAD_SMALL_BLOCK_SIZE;
            *tt = * tt + 1;
        }
    }

} 

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int accessOrder[ACCESS_NUMBER_GROUP];
    FILE *file;

    //open the file
    file = fopen("accessOrder.txt", "r");
    if (file == NULL) {
        printf("The file cannot be open\n");
        return 1;
    }
    // read the Access order
    for (int i = 0; i < ACCESS_NUMBER_GROUP; i++) {
        fscanf(file, "%d", &accessOrder[i]);
    }
    // close
    fclose(file);

    //define the order of inner group 
    int innerGroupOrder[ACCESS_GROUP_SIZE] = {0, 1, 2, 4, 8, 16, 31};
    int* d_innerGroupOrder;
    cudaMalloc(&d_innerGroupOrder, ACCESS_GROUP_SIZE * sizeof(int));
    cudaMemcpy(d_innerGroupOrder, innerGroupOrder, ACCESS_GROUP_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    int* d_accessOrder;
    cudaMalloc(&d_accessOrder, ACCESS_NUMBER_GROUP * sizeof(int));
    cudaMemcpy(d_accessOrder, accessOrder, ACCESS_NUMBER_GROUP * sizeof(int), cudaMemcpyHostToDevice);

    long long N = 1024LL*1024LL*1024LL*120LL;
    char *startAddress;
   
    cudaMallocManaged(&startAddress, N*sizeof(char));
    memset(startAddress, 0, N*sizeof(char));

    char* warpStartAddress[WARP_NUMBER*ACCESS_NUMBER_GROUP];
    warpStartAddress[0] = startAddress;
    for(int i = 1; i < WARP_NUMBER*ACCESS_NUMBER_GROUP; i++)
    {
        warpStartAddress[i] = warpStartAddress[i-1] + THREAD_LARGE_BLOCK*32;
    }
    char** d_warpStartAddress;
    cudaMalloc(&d_warpStartAddress, WARP_NUMBER*ACCESS_NUMBER_GROUP * sizeof(char*));
    cudaMemcpy(d_warpStartAddress, warpStartAddress, WARP_NUMBER*ACCESS_NUMBER_GROUP * sizeof(char*), cudaMemcpyHostToDevice);

    int indexLane[32] = {0,1,2,4, 6, 8, 9, 10, 12, 16, 17, 18,20,26,29, 30,25,23,21,19,15,14,13,11,7,5,3,28,27,24,31,22};
    int* d_indexLane;
    cudaMalloc(&d_indexLane, 32 * sizeof(int));
    cudaMemcpy(d_indexLane, indexLane, 32 * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    accessOver<<<BLOCK_NUMBER, BLOCK_SIZE>>>(d_innerGroupOrder, d_accessOrder, d_warpStartAddress, d_indexLane);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}