#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "MMUOnCPU.hpp"
#include <cuda_runtime.h>

#include <time.h>

/* 
Version 1.1 
In the host, memoryManagement calls the function, allocateMem(), to allocate memory

Version 1.2
add block sychronization
reconstruct the code. users could call the allocateThr and freeThr to call it.
All the scheduling for allocation and free is in the host.
CPU use a number of thread to deal with the allocation and free requests from GPU. ALLOCATIONMANAGEMENTTHREADNUMBER is used for defining how many threads.
Supports multi-kernel allocation. For each kernel, the host allocate an independent memory meta data to manage it and a thread to deal with.

compiler command
nvcc -rdc=true -arch=sm_80 randomAllocationLarge.cu MMUOnCPU.cu -lpthread
*/


#define ALL_FREE_TIMES 8000
#define PORTION_PER_THREAD 500
#define MIN_ALLO_SIZE_RANDOM 1024
#define MAX_ALLO_SIZE_RANDOM 262143

typedef struct Node {
    int data;
    struct Node* next;
} Node;

// Function prototypes
Node* createNode(int data);

// Function to check if a value exists in the list
int existsInList(Node* head, int data);

// Function to insert a new unique random number into the list
int insertUniqueRandom(Node** head);

// Function to delete a random node from the list
int deleteRandom(Node** head, int length);

void generateSeq(bool* isAllocate, int* freeBlock, int* allocateSizeRandom);


/* -- test for version 1.2 --*/
__global__ void waitForHostAndContinue(struct MMUOnTransfer** pMMUOnTransfer, bool* d_isAllocate, int* d_freeBlock, int* d_allocateSizeRandom)
{
    int* allocationPointers[PORTION_PER_THREAD];
    // int* a = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // if(threadIdx.x == 0)
    // {
    //     printf("step i free %p\n", a);
    // }
    // __syncthreads();
    // int* b = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* c = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* d = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* e = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // freeThr(d, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // __syncthreads();
    // freeThr(c, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // __syncthreads();
    // freeThr(e, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // __syncthreads();
    // int* f = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* f1 = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* f2 = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // __syncthreads();
    // int* f3 = (int*)allocateThr(1024, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // if(threadIdx.x == 0)
    // {
    //     printf("step i free %p\n", f);
    // }
    // freeThr(b, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // __syncthreads();
    // freeThr(c, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);

    // int* a = (int*)allocateThr(1024, pMMUOnTransfer, isWarpBranchAllocate);

    for(int i = 0; i < ALL_FREE_TIMES; i++)
    {
        if(d_isAllocate[i] == 1)
        {
            allocationPointers[d_freeBlock[i]] = (int*)allocateThr(d_allocateSizeRandom[i], pMMUOnTransfer);
            // if(threadIdx.x == 0 && blockIdx.x == 0)
            // {
            //     printf("step i allocate %d  %d\n", i, d_allocateSizeRandom[i]);
            // }
        }else
        {
            // deallocateMemoryPoolGPURandomlyStack(allocationPointers[d_freeBlock[i]], &s);
            freeThr(allocationPointers[d_freeBlock[i]], pMMUOnTransfer);
            // if(threadIdx.x == 0 && blockIdx.x == 0)
            // {
            //     printf("step i free %d  %p\n", i, allocationPointers[d_freeBlock[i]]);
            // }
        }
        __syncthreads();
    }


    // //allocate call
    // int tid =  threadIdx.x;
    // int* a = (int*)allocateThr(4, pMMUOnTransfer, d_sizetmpAllocate, &store, d_sizeTotalallocate);
    // // *a = tid + 20;
    // // if(tid == 5)
    // // {
    // //     pMMUOnTransfer -> sizeAllocate[blockIdx.x] = *a;
    // // }
    
    // // if(tid == 0)printf("a\n");
    // // int* b = (int*)allocateThr(10, pMMUOnTransfer);
    // // *b = tid + 30;
    // if(tid != 1)
    // {
    //     freeThr(a, pMMUOnTransfer, &store, d_sizeTotalallocate, d_sizetmpAllocate);
    // }
    
}

int main() 
{
    //generate a allocate or free sequence in a random order
    bool isAllocate[ALL_FREE_TIMES];
    int freeBlock[ALL_FREE_TIMES];
    int allocateSizeRandom[ALL_FREE_TIMES];

    //device varaiables
    bool *d_isAllocate;
    int *d_freeBlock;
    int *d_allocateSizeRandom;

    // Allocate memory on the device
    cudaMalloc(&d_isAllocate, ALL_FREE_TIMES * sizeof(bool));
    cudaMalloc(&d_freeBlock, ALL_FREE_TIMES * sizeof(int));
    cudaMalloc(&d_allocateSizeRandom, ALL_FREE_TIMES * sizeof(int));

    //init allocatio sequence list
    generateSeq(isAllocate, freeBlock, allocateSizeRandom);

    // Copy the host array to the device
    cudaMemcpy(d_isAllocate, isAllocate, ALL_FREE_TIMES * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_freeBlock, freeBlock, ALL_FREE_TIMES * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_allocateSizeRandom, allocateSizeRandom, ALL_FREE_TIMES * sizeof(int), cudaMemcpyHostToDevice);
    
    /* -- init meta -- */
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
    waitForHostAndContinue<<<BLOCKNUMBER, BLOCKSIZE, 0, stream>>>(d_pMMUOnTransfer, d_isAllocate, d_freeBlock, d_allocateSizeRandom);
    
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

    // Free device memory
    cudaFree(d_isAllocate);
    cudaFree(d_freeBlock);

    
    // int* a = (int*)memoryManagement.allocateMem(sizeof(int)*1024*2);
    // int* b = (int*)memoryManagement.allocateMem(sizeof(int));
    // int* c = (int*)memoryManagement.allocateMem(sizeof(int)*1024*4);
    // memoryManagement.freeMem(a);
    // memoryManagement.freeMem(b);
    // memoryManagement.freeMem(c);
    // int* d = (int*)memoryManagement.allocateMem(sizeof(int)*1024*4);
    // memoryManagement.freeMem(d);

    // int* e = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*8);
    
    // int* f = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*10);
    // int* g = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*16);
    // memoryManagement.freeMem(f);
    // int* h = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*16);
    // int* i = (int*)memoryManagement.allocateMem(sizeof(int)*1024*1024*9);

    
    // memoryManagement.freeMem(g);
    
    // memoryManagement.freeMem(h);
    
    // memoryManagement.freeMem(e);
    // memoryManagement.freeMem(i);

}

//implementation
Node* createNode(int data) 
{
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (newNode == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    newNode->data = data;
    newNode->next = NULL;
    return newNode;
}

int existsInList(Node* head, int data) 
{
    Node* current = head;
    while (current != NULL) {
        if (current->data == data) {
            return 1; // Data found
        }
        current = current->next;
    }
    return 0; // Data not found
}

int insertUniqueRandom(Node** head) 
{
    while(1)
    {
        int randomData = rand() % PORTION_PER_THREAD; // Random number between 0 and PORTION_PER_THREAD
        if (!existsInList(*head, randomData)) {
            Node* newNode = createNode(randomData);
            newNode->next = *head;
            *head = newNode;
            // printf("insert %d\n", randomData);
            return randomData;
        } 
    }
}

int deleteRandom(Node** head, int length) 
{
    int randomIndex = rand() % length;
    Node* prev = NULL;
    Node* current = *head;

    if (randomIndex == 0) { // Delete the head
        *head = current->next;
        int valueD = current -> data;
        free(current);
        // printf("Deleted %d \n", valueD);
        return valueD;
    }

    for (int i = 0; i < randomIndex; i++) {
        prev = current;
        current = current->next;
    }

    prev->next = current->next;
    int valueD = current -> data;
    free(current);
    // printf("Deleted %d\n", valueD);
    return valueD;
}

void generateSeq(bool* isAllocate, int* freeBlock, int* allocateSizeRandom)
{
    srand(time(NULL));
    Node* head = NULL;
    int nodeLength = 0;

    for(int i = 0; i < ALL_FREE_TIMES; i++)
    {
        //if the allocated list is empty, allocate one.
        if(nodeLength == 0)
        {
            freeBlock[i] = insertUniqueRandom(&head);
            isAllocate[i] = 1;
            nodeLength ++;
            allocateSizeRandom[i] = rand() % (MAX_ALLO_SIZE_RANDOM - MIN_ALLO_SIZE_RANDOM + 1) + MIN_ALLO_SIZE_RANDOM;
            //printf("gene %d\n", allocateSizeRandom[i]);
            continue;
        }

        //if the allocated list is full, delete one.
        if(nodeLength == PORTION_PER_THREAD)
        {
            isAllocate[i] = 0;
            freeBlock[i] = deleteRandom(&head, nodeLength);
            nodeLength --;
            continue;
        }

        //allocate or free randomly
        int isallocate = rand() % 2;
        //if allocate
        if(isallocate == 1)
        {
            freeBlock[i] = insertUniqueRandom(&head);
            isAllocate[i] = 1;
            nodeLength ++;
            allocateSizeRandom[i] = rand() % (MAX_ALLO_SIZE_RANDOM - MIN_ALLO_SIZE_RANDOM + 1) + MIN_ALLO_SIZE_RANDOM;
            //printf("gene %d\n", allocateSizeRandom[i]);
        }else
        {
            isAllocate[i] = 0;
            freeBlock[i] = deleteRandom(&head, nodeLength);
            nodeLength --;
        }
    }
}

