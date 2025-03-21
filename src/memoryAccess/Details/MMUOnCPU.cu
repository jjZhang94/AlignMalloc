#include "MMUOnCPU.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include <curand_kernel.h>

#include <unistd.h>

/* -- version 1.3*/
// Memory pool structure
// __device__ char memoryPool[POOL_SIZE];
// __device__ int allocationMapGPU[TOTAL_PORTIONS]; // 0 = free, 1 = allocated
/* -- End -- version 1.3*/

/* -- Implementation of memory management with UVMManagement -- */
UVMManagement::UVMManagement(void * startAddressIn)
{
    //init start address
    startAddress = startAddressIn;

    //init bitmap
    initTree();

    //init hash tables
    initializeHashTable();
}

UVMManagement::~UVMManagement()
{
    freeHashTable();
}

void* UVMManagement::allocateMem(unsigned int sizeAllocate)
{
    //get the size of continuous pages
    unsigned int pages = ceil(float((sizeAllocate) / VABLOCK_SIZE));
    //printf("SSS %d\n", pages); 
    //if just need only one page, just find a idle block
    if (pages == 1)
    {  
        unsigned int targetBlockIndex = findMinOne();
        setNode2Zero(targetBlockIndex);
        insertHashTable(targetBlockIndex, pages);
        return (void*)((char*)startAddress + targetBlockIndex*VABLOCK_SIZE);
    }

    //more than one pages
    unsigned int startBlockIndex = findFirstFitKConsecutiveOnes(pages);
    // printf("startBlockIndex %d\n", startBlockIndex);
    setZeroConsecutiveKOnesOptimized(startBlockIndex, pages);

    //update block-size table
    insertHashTable(startBlockIndex, pages);

    return (void*)((char*)startAddress + startBlockIndex*VABLOCK_SIZE);

}

void UVMManagement::freeMem(void* addressIn)
{
    //get the size of blockID
    unsigned int offsetAddress = (unsigned int)((char*)addressIn - (char*)startAddress);
    unsigned int blockID = offsetAddress / VABLOCK_SIZE;

    //search the blockID and its size
    unsigned pageNumber = findHashTable(blockID);
    //if just free only one page, just clear the block
    if(pageNumber == 1)
    {
        setNode2One(blockID);  
    }else
    {
        setOneConsecutiveKZeros(blockID, pageNumber);
    }
    
    deleteHashTable(blockID);
    return;
      
}

void UVMManagement::initTree()
{
    for (int i = 0; i < TOTAL_NODES; i++) {
        tree[i] = false;
    }
}

int UVMManagement::parent(int idx) {
    return (idx - 1) / 2;
}

int UVMManagement::leftChild(int idx) {
    return 2 * idx + 1;
}

int UVMManagement::rightChild(int idx) {
    return 2 * idx + 2;
}

//a middle function to update the node for recursion
void UVMManagement::updateTree(int idx, bool value) {
    // Start from the given index, update the value and propagate up
    tree[idx] = value;
    while (idx > 0) {
        int p = parent(idx);
        tree[p] = tree[leftChild(p)] | tree[rightChild(p)];
        idx = p;
    }
}

//set the leaf index to 1 and update the tree nodes
void UVMManagement::setNode2One(int leaf_idx) {
    // Convert leaf index to the corresponding index in the array
    int node_idx = NUM_LEAVES - 1 + leaf_idx;
    updateTree(node_idx, true);
}

//set the leaf index to 0 and update the tree nodes
void UVMManagement::setNode2Zero(int leaf_idx) {
    int node_idx = NUM_LEAVES - 1 + leaf_idx;
    updateTree(node_idx, false);
}

void UVMManagement::printTree() {
    for (int i = 0; i < TOTAL_NODES; i++) {
        printf("%d: %d\n", i, tree[i]);
    }
}

//find the 1 in first fit strategy
int UVMManagement::findMinOne() {
    if (tree[0] == false) { // If the root is 0, there are no 1s in the tree
        return -1;
    }
    
    int idx = 0; // Start from the root
    while (idx < NUM_LEAVES - 1) { // Continue until reaching leaf level
        int leftChild = 2 * idx + 1;
        int rightChild = 2 * idx + 2;
        
        // Check if the left child is non-zero
        if (tree[leftChild]) {
            idx = leftChild;
        } else if (tree[rightChild]) { // If left is zero, go right
            idx = rightChild;
        }
    }
    
    // At this point, idx is the index of the leaf with the smallest 1
    return idx - (NUM_LEAVES - 1); // Convert array index to leaf index
}

//find the next 1 element after the element 1
int UVMManagement::findSuccessor(int leaf_idx) {
    int node_idx = NUM_LEAVES - 1 + leaf_idx; // Convert leaf index to node index
    int p = parent(node_idx);
    
    while (node_idx != leftChild(p) || !tree[rightChild(p)]) {
        if(p == 0) return -1;
        node_idx = p;
        p = parent(p);
    }

    // Find the minimum 1 in the right subtree
    node_idx = rightChild(p);
    while (node_idx < NUM_LEAVES - 1) { // Traverse until a leaf is reached
        if (tree[leftChild(node_idx)]) {
            node_idx = leftChild(node_idx);
        } else if (tree[rightChild(node_idx)]) {
            node_idx = rightChild(node_idx);
        } else {
            break; // No non-zero nodes found further
        }
    }
    return node_idx - (NUM_LEAVES - 1); // Convert back to leaf index
}

//find the k consecutive 1 elements in first fit strategy 
int UVMManagement::findFirstFitKConsecutiveOnes(int k) {
    int startIndex = findMinOne();
    if (startIndex == -1) return -1; // No ones at all

    int count = 1;
    int currentIndex = startIndex;

    while (count < k) {
        if (currentIndex + 1 < NUM_LEAVES && tree[NUM_LEAVES - 1 + currentIndex + 1]) {
            // Check next index directly
            currentIndex++;
            count++;
        } else {
            // Find the next 1 using successor function
            int nextIndex = findSuccessor(currentIndex);
            if (nextIndex == -1) return -1; // No more ones available

            // Restart count if not consecutive
            currentIndex = nextIndex;
            startIndex = currentIndex;
            count = 1;
        }
    }

    return startIndex; // Return the start index of the first fitting k consecutive ones
}

void UVMManagement::setZeroConsecutiveKOnesOptimized(int startIdx, int k) {
    int firstNodeIdx = NUM_LEAVES - 1 + startIdx; //translate into the index of tree nodes
    int lastNodeIdx = firstNodeIdx + k - 1;

    //set the zero to leaf nodes
    for (int i = firstNodeIdx; i <= lastNodeIdx; i++) {
        tree[i] = false;
    }

    //update the parents
    int lowestParentIdx = parent(firstNodeIdx);
    int highestParentIdx = parent(lastNodeIdx);

    
    for (int i = lowestParentIdx; i >= 0; i--) {
        tree[i] = tree[leftChild(i)] | tree[rightChild(i)];
        if (i == 0) break; //end in the root 
    }
}

void UVMManagement::setOneConsecutiveKZeros(int startIdx, int k) {
    int firstNodeIdx = NUM_LEAVES - 1 + startIdx; //translate into the index of tree nodes
    int lastNodeIdx = firstNodeIdx + k - 1;

    //set the 1 to leaf nodes
    for (int i = firstNodeIdx; i <= lastNodeIdx; i++) {
        tree[i] = true;
    }

    //update the parents
    int lowestParentIdx = parent(firstNodeIdx);
    int highestParentIdx = parent(lastNodeIdx);

    for (int i = lowestParentIdx; i >= 0; i = parent(i)) {
        tree[i] = tree[leftChild(i)] | tree[rightChild(i)];
        if (i == 0) break; //end in the root 
    }
}

unsigned int UVMManagement::hash(unsigned int key)
{
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = ((key >> 16) ^ key) * 0x45d9f3b;
    key = (key >> 16) ^ key;
    return key & (HASHTABLE_BLOCKIDSIZE_SIZE - 1); // TABLE_SIZE is 4096
}

void UVMManagement::initializeHashTable()
{
    for (int i = 0; i < HASHTABLE_BLOCKIDSIZE_SIZE; ++i) {
        buckets[i] = NULL;
    }
}

void UVMManagement::insertHashTable(unsigned int key, unsigned int value)
{
    unsigned int idx = hash(key);
    BlockIDSizeNode* newNode = (BlockIDSizeNode*)malloc(sizeof(BlockIDSizeNode));

    newNode->pair.blockID = key;
    newNode->pair.size = value;

    // Insert at the beginning of the chain (bucket)
    newNode->next = buckets[idx];
    buckets[idx] = newNode;
}

unsigned int UVMManagement::findHashTable(unsigned int key)
{
    unsigned int idx = hash(key);
    unsigned int returnValue;
    BlockIDSizeNode* current = buckets[idx];
    while (current != NULL) {
        if (current->pair.blockID == key)
        {
            returnValue = current->pair.size;
            return returnValue;
        }
        current = current->next;
    }
    return 0;
}

void UVMManagement::deleteHashTable(unsigned int key)
{
    unsigned int idx = hash(key);
    BlockIDSizeNode *current = buckets[idx];
    BlockIDSizeNode *prev = NULL;
    while (current != NULL) {
        if (current->pair.blockID == key) {
            if (prev == NULL) {
                buckets[idx] = current->next;
            } else {
                prev->next = current->next;
            }
            free(current);
            return;
        }
        prev = current;
        current = current->next;
    }
}

void UVMManagement::freeHashTable()
{
    for (int i = 0; i < HASHTABLE_BLOCKIDSIZE_SIZE; ++i) {
        BlockIDSizeNode* current = buckets[i];
        while (current != NULL) {
            BlockIDSizeNode* temp = current;
            current = current->next;
            free(temp);
        }
    }
}
/* -- End Implementation of memory management with UVMManagement -- */

/* -- Implementation of memory management -- */
MemoryManagement::MemoryManagement()
{
    //init startAddress
    long long size = UVM_SIZE;
    // size = size * 4;
    cudaMallocManaged(&UVMStartAddress, size);

    //init class BitmapManagement and LinkedListManagement
    pUVMManagement = new UVMManagement(UVMStartAddress);
}

MemoryManagement::~MemoryManagement()
{
    delete pUVMManagement;
    cudaFree(UVMStartAddress);
}

void MemoryManagement::allocateMem(struct MMUOnTransfer* pMMUOnTransfer)
{
    pMMUOnTransfer -> addressAllocate = pUVMManagement->allocateMem(pMMUOnTransfer -> sizeAllocate);
}

void MemoryManagement::freeMem(struct MMUOnTransfer* pMMUOnTransfer)
{
    pUVMManagement->freeMem(pMMUOnTransfer -> addressFree);
}

void* MemoryManagement::getUVMStartAddress()
{
    return UVMStartAddress;
}

/* -- End -- Implementation of memory management -- */

/* -- allocation function*/
//launch threads to deal with blocks allocation and free
void* threadAllocation(void* arg)
{
    thread_args* args = (thread_args*)arg;
    
    //create threads for block allocations
    if(WARPNUMBER < ALLOCATIONMANAGEMENTTHREADNUMBER)
    {
        // if the number of threads smaller than blocknumber
        // Create and start threads
        pthread_t threads[WARPNUMBER];
        threadBlockAllocations pthreadBlockAllocations[WARPNUMBER];
        for(int i = 0; i < WARPNUMBER; ++i) 
        {
            pthreadBlockAllocations[i].pMMUOnTransfer = args -> pMMUOnTransfer;
            pthreadBlockAllocations[i].should_exit = args -> should_exit;
            pthreadBlockAllocations[i].pmemoryManagement = args -> pmemoryManagement[i];
            pthreadBlockAllocations[i].start = i;
            pthreadBlockAllocations[i].end = i + 1;

            pthread_create(&threads[i], NULL, blockAllocationThr, &pthreadBlockAllocations[i]);
        }

        for (int i = 0; i < WARPNUMBER; ++i) {
            pthread_join(threads[i], NULL);
        }
    }else{
        pthread_t threads[ALLOCATIONMANAGEMENTTHREADNUMBER];
        threadBlockAllocations pthreadBlockAllocations[ALLOCATIONMANAGEMENTTHREADNUMBER];

        for (int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; ++i) {
            pthreadBlockAllocations[i].pMMUOnTransfer = args -> pMMUOnTransfer;
            pthreadBlockAllocations[i].should_exit = args -> should_exit;
            pthreadBlockAllocations[i].pmemoryManagement = args -> pmemoryManagement[i];
            pthreadBlockAllocations[i].start = i * (WARPNUMBER / ALLOCATIONMANAGEMENTTHREADNUMBER);
            pthreadBlockAllocations[i].end = (i + 1) * (WARPNUMBER / ALLOCATIONMANAGEMENTTHREADNUMBER);
            if(i == (ALLOCATIONMANAGEMENTTHREADNUMBER - 1))
            {
                pthreadBlockAllocations[i].end = WARPNUMBER;
            }
            // printf("start %d. end:%d\n", pthreadBlockAllocations[i].start, pthreadBlockAllocations[i].end);
            pthread_create(&threads[i], NULL, blockAllocationThr, &pthreadBlockAllocations[i]);
        }

        for (int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; ++i) {
            pthread_join(threads[i], NULL);
        }
    }
    return NULL;

    // while(1)
    // {
    //     while (args->pMMUOnTransfer->syncFlag>0 && !args->should_exit) {
    //     }
    //     if(args->should_exit)break;
        
    //     //allocating calculate
    //     args->pmemoryManagement->allocateMem(args->pMMUOnTransfer);
        
    //     //after finishing, notify GPU to continue
    //     args->pMMUOnTransfer -> syncFlag = 1;
    // }
    // pthread_exit(NULL);
}

//each thread deal with block allocations
void* blockAllocationThr(void* arg)
{
    threadBlockAllocations* data = (threadBlockAllocations*)arg;
    
    while (true) {
        for (int i = data->start; i < data->end; ++i) {
            //chek if there are allocation
            if ((data->pMMUOnTransfer[i])->syncFlag == 0){
                //allocating calculate
                // printf("bb\n");
                data -> pmemoryManagement->allocateMem(data->pMMUOnTransfer[i]);
                // printf("allocate %d %p\n", i, (data->pMMUOnTransfer[i])->addressAllocate);
                // Perform some calculation...
                // printf("aa\n");
                //after finishing, notify GPU to continue
                (data -> pMMUOnTransfer[i]) -> syncFlag = 1;
            }
            
            //chek if there are free
            if ((data->pMMUOnTransfer[i])->syncFlag == 2){
                //allocating calculate
                
                data -> pmemoryManagement->freeMem(data->pMMUOnTransfer[i]);
                // Perform some calculation...
                
                //after finishing, notify GPU to continue
                (data -> pMMUOnTransfer[i]) -> syncFlag = 1;

            }
        }

        //when kernel finishes, checking is over
        if(*(data -> should_exit))
        {
            break;
        }
    }

    pthread_exit(NULL);
}

//after launch kernel
void aLaunchKernel(thread_args* args, cudaStream_t stream)
{
    // //thread launch
    // pthread_t thread_id;

    // // Initialize thread arguments
    // thread_args args;
    //  = { .pMMUOnTransfer = pMMUOnTransfer}; 
    // int ii = 0;
    // args.should_exit = &ii;
    // for(int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; i++)
    // {
    //     args.pmemoryManagement[i] = memoryManagement[i];
    // }

    // // Create the thread
    // pthread_create(&thread_id, NULL, threadAllocation, &args);
    // if (pthread_create(&thread_id, NULL, threadAllocation, &args)) {
    //     fprintf(stderr, "Error creating thread\n");
    // }
    
    // Wait for the kernel in this stream to complete
    cudaError_t error = cudaStreamSynchronize(stream);
    
    // Signal the thread to exit
    *(args -> should_exit) = 1;

    // Wait for the thread to finish
    // pthread_join(thread_id, NULL);

    // if (pthread_join(thread_id, NULL)) {
    //     fprintf(stderr, "Error joining thread\n");
    // }

    cudaStreamDestroy(stream);
}

//before launch kernel, init it
void initAllocationStru(MemoryManagement* memoryManagement[], struct MMUOnTransfer **pMMUOnTransfer, pthread_t* thread_id, thread_args* args, int* should_exit)
{    
    //create MemoryManagement
    for(int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; i++)
    {
        memoryManagement[i] = new MemoryManagement();
    }

    for(int i = 0; i < WARPNUMBER; i++)
    {
        pMMUOnTransfer[i] -> sizeAllocate = 0;
        pMMUOnTransfer[i] -> syncFlag = 1;
        pMMUOnTransfer[i] -> addressAllocate = NULL;
        pMMUOnTransfer[i] -> addressFree = NULL;
    }

    // Initialize thread arguments
    args ->  pMMUOnTransfer = pMMUOnTransfer; 
    args -> should_exit = should_exit;
    for(int i = 0; i < ALLOCATIONMANAGEMENTTHREADNUMBER; i++)
    {
        args -> pmemoryManagement[i] = memoryManagement[i];
    }

    if (pthread_create(thread_id, NULL, threadAllocation, args)) {
        fprintf(stderr, "Error creating thread\n");
    }
}
/* -- End -- allocation function*/

//allocation on Each thread
__device__ void* allocateThr(size_t allocateSize, struct MMUOnTransfer** pMMUOnTransfer, struct MetaDataAllocationGPU* d_MetaDataAllocationGPU)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = idx / warpSize;  // Warp index
    int laneId = threadIdx.x % warpSize;  // Lane index within the warp

    //reduce funciton to get sum
    int value = allocateSize;

    // Warp-wide reduction using __reduce_add_sync to get sum of allocateSize
    unsigned mask = 0xffffffff; // Full warp mask
    int sum = __reduce_add_sync(mask, value);

    if (laneId == 0)
    {
        (pMMUOnTransfer[warpId])->sizeAllocate = sum;
        (pMMUOnTransfer[warpId]) -> syncFlag = 0;

    } 

    while (atomicAdd(&((pMMUOnTransfer[warpId]) -> syncFlag), 0) != 1) 
    {
    }
    
    void* tmpStore =  (void*)((pMMUOnTransfer[warpId]) -> addressAllocate);

    // if the size is different 
    //check if the allocation size is same.
    // Use __match_all_sync to check if all active threads have the same value
    // int isMatch = 0;
    // __match_all_sync(mask, value, &isMatch);

    __syncthreads();

    if(laneId == 0)
    {
        (pMMUOnTransfer[warpId]) -> addressAllocate = NULL;
        
        //update the metadata in devices
        d_MetaDataAllocationGPU[warpId].sizeAllocation = allocateSize;
        d_MetaDataAllocationGPU[warpId].startAddress = (char*)tmpStore;
    }

    return tmpStore;
}


__device__ void freeThr(void* freeAddress, struct MMUOnTransfer** pMMUOnTransfer)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = idx / warpSize;  // Warp index
    int laneId = threadIdx.x % warpSize;  // Lane index within the warp

    if(laneId == 0)
    {
        (pMMUOnTransfer[warpId]) -> addressFree = freeAddress;
        (pMMUOnTransfer[warpId]) -> syncFlag = 2;

        while (atomicAdd(&((pMMUOnTransfer[warpId]) -> syncFlag), 0) == 2) 
        {
        }
    }

    __syncthreads();
   
}

/*-- memory access*/
__device__ void* addressAccess(void* inputAddress, struct MetaDataAllocationGPU* d_MetaDataAllocationGPU)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = idx / warpSize;
    int laneId = threadIdx.x % warpSize; 

    char* stratAddressThread = d_MetaDataAllocationGPU[warpId].startAddress;

    unsigned int offsetAddress = (char*)inputAddress - stratAddressThread;
    int offsetBlock = offsetAddress/WARPGAP;
    void* returnPointer = (char*)inputAddress + (offsetBlock * warpSize + laneId - offsetBlock) * WARPGAP;
    return returnPointer;
}

__device__ void memcpyRead(void* inputAddress, void* outputAddress, unsigned int length, struct MetaDataAllocationGPU* d_MetaDataAllocationGPU)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int warpId = idx / warpSize;
    int laneId = threadIdx.x % warpSize; 

    char* stratAddressThread = d_MetaDataAllocationGPU[warpId].startAddress;
    unsigned int offsetAddress = (char*)inputAddress - stratAddressThread;

    int offsetBlock = offsetAddress/WARPGAP;
    char* returnPointer = (char*)inputAddress + (offsetBlock * warpSize + laneId - offsetBlock) * WARPGAP;

    int startGroup = offsetAddress / WARPGAP;
    int startOffset = offsetAddress % WARPGAP;
    int endGroup = (offsetAddress + length - 1) / WARPGAP;
    int endOffset = (offsetAddress + length - 1) % WARPGAP;

    unsigned int lengthTotal = 0;
    for (int i = startGroup; i <= endGroup; i++) {
        int startOffsetCopy = 0;
        int endOffsetCopy = WARPGAP - 1;
        if(i == startGroup)
        {
            startOffsetCopy = startOffset;
        }
        if(i == endGroup)
        {
            endOffsetCopy = endOffset;
        }

        length = endOffsetCopy + 1 - startOffsetCopy;

        memcpy(outputAddress, returnPointer + lengthTotal, length);
        lengthTotal = lengthTotal + length + WARPGAP * (warpSize - 1);
    }
}
/*-- End memory access*/
/* -- End user library in GPU*/