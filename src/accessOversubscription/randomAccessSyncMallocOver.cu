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

//command line nvcc -rdc=true -arch=sm_80 randomAccessSyncMallocOver.cu

__global__ void accessOver(int* d_innerGroupOrder, int* d_accessOrder, char** d_threadStartAddress)
{
    //   clock_t cycles = 1000;   //waiting cycles
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    char* startAddress = d_threadStartAddress[index];

    for(int i = 0; i < ACTUALACCESSTIME; i++)
    {
        char *tmpStartAddress = startAddress + THREAD_LARGE_BLOCK * d_accessOrder[i];
        for(int j = 0; j < ACCESS_GROUP_SIZE; j++)
        {
            //according to inner group order fetch
            char *tt = tmpStartAddress + d_innerGroupOrder[j] * THREAD_SMALL_BLOCK_SIZE;
            *tt = *tt + 1;
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

    char* threadStartAddress[THREADNUMBER];
    threadStartAddress[0] = startAddress;
    for(int i = 1; i < THREADNUMBER; i++)
    {
        threadStartAddress[i] = threadStartAddress[i-1] + THREAD_GAP;
    }
    char** d_threadStartAddress;
    cudaMalloc(&d_threadStartAddress, THREADNUMBER * sizeof(char*));
    cudaMemcpy(d_threadStartAddress, threadStartAddress, THREADNUMBER * sizeof(char*), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    accessOver<<<BLOCK_NUMBER, BLOCK_SIZE>>>(d_innerGroupOrder, d_accessOrder, d_threadStartAddress);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f milliseconds\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}