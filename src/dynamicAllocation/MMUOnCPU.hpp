#ifndef EXTERNAL_H
#define EXTERNAL_H

// some parameters for optimizaiton
#define ALLOCATIONMANAGEMENTTHREADNUMBER  16 //there are a number of threads for dealing with allocaiton from GPU blocks

//Warp allocation division
#define WARPALLOCATIONMAXTIMES 100

//user define
#define BLOCKNUMBER 16
#define BLOCKSIZE 1024

#define WARPSIZE 32
#define WARPNUMBER (BLOCKSIZE * BLOCKNUMBER / WARPSIZE)

#define VABLOCK_SIZE (2*1024*1024)  // size is 2MB
#define UVM_SIZE  (1024LL*1024LL*1024LL*16LL)    //16GB
#define NUM_LEAVES (512*16)
#define TOTAL_NODES (2*NUM_LEAVES - 1)

#define HASHTABLE_BLOCKIDSIZE_SIZE 4096

#define MAX_SUBTREE_SIZE 4095

//hash is used for store the size of allocation
typedef struct {
    unsigned int blockID;
    unsigned int size;
} BlockIDSize;

typedef struct BlockIDSizeNode {
    BlockIDSize pair;
    struct BlockIDSizeNode* next;
} BlockIDSizeNode;

class UVMManagement
{  
    private:
        void* startAddress; //a pointer pointing to unfied memory allocated
        bool tree[TOTAL_NODES];  // a full binaray tree to track memory blocks
        BlockIDSizeNode* buckets[HASHTABLE_BLOCKIDSIZE_SIZE];  //used for storing the blockID and their size
        
        /*————Definitions of Functions */
        void initTree(); // init tree
        // get the parent
        int parent(int idx);
        // get the left child
        int leftChild(int idx);
        // get the right child
        int rightChild(int idx);
        //a middle function to update the node for recursion
        void updateTree(int idx, bool value);
        //set the leaf index to 1 and update the tree nodes
        void setNode2One(int leaf_idx);
        //set the leaf index to 0 and update the tree nodes
        void setNode2Zero(int leaf_idx);
        //print the tree nodes in sequence traversal
        void printTree();
        //find the 1 in first fit strategy
        int findMinOne();
        //find the next 1 element after the element 1
        int findSuccessor(int leaf_idx);
        //find the k consecutive 1 elements in first fit strategy 
        int findFirstFitKConsecutiveOnes(int k);
        //set k consecutive ones to zeros
        void setZeroConsecutiveKOnesOptimized(int startIdx, int k);
        //set k consecutive zeros to ones
        void setOneConsecutiveKZeros(int startIdx, int k);

        //get the value from index
        bool getValue(int index);
        
        /*--hash table functions for bockID*/
        //hash function
        unsigned int hash(unsigned int key);
        //init hash table
        void initializeHashTable();
        //insertion of hash table
        void insertHashTable(unsigned int key, unsigned int value);
        //Find of hash table, and return the size
        unsigned int findHashTable(unsigned int key);
        //Delete of hash table
        void deleteHashTable(unsigned int key);
        //free Entire Hash Table
        void freeHashTable();

    public:
        UVMManagement(void * startAddressIn);
        ~UVMManagement();
        
        //Allocation new memory with bitmaps
        void* allocateMem(unsigned int sizeAllocate);
        //free memory with bitmaps
        void freeMem(void* addressIn);

};

/*-- End -- Version 1.3.5*/

/*-- Define data structure for memory management--*/
class MemoryManagement
{
    private:
        void* UVMStartAddress;
        UVMManagement *pUVMManagement;
    public:
        MemoryManagement();
        ~MemoryManagement();

        void* getUVMStartAddress();

        void allocateMem(struct MMUOnTransfer* pMMUOnTransfer);
        void freeMem(struct MMUOnTransfer* pMMUOnTransfer);
};
/*-- End -- Define data structure for memory management--*/


/* --  Define data structure for Transferring between CPU and GPU--*/
typedef struct MMUOnTransfer
{
    // void* bitmapStartAddress;
    // void* linkedListStartAddress;
    int syncFlag;
    unsigned int sizeAllocate;
    void* addressAllocate;
    void* addressFree;
} MMUOnTransfer;

/* -- End --  Define data structure for Transferring between CPU and GPU--*/

//a struct message for a allocation thread arg
typedef struct {
    MemoryManagement *pmemoryManagement[ALLOCATIONMANAGEMENTTHREADNUMBER];
    struct MMUOnTransfer **pMMUOnTransfer;
    int *should_exit; // Flag to indicate the thread should exit
} thread_args;

//a struct message for deal with some block allocations
typedef struct {
    int start;
    int end;
    // volatile int should_exit; // Flag to indicate the thread should exit
    struct MMUOnTransfer **pMMUOnTransfer;
    int *should_exit; // Flag to indicate the thread should exit
    MemoryManagement *pmemoryManagement;
} threadBlockAllocations;


/* -- allocation function*/
//allocation function as a thread
void* threadAllocation(void* arg);
//each thread deal with block allocations
void* blockAllocationThr(void* arg);
//after launch kernel
void aLaunchKernel(thread_args* args, cudaStream_t stream);
//before launch kernel, init it
void initAllocationStru(MemoryManagement* memoryManagement[], struct MMUOnTransfer **pMMUOnTransfer, pthread_t* thread_id, thread_args* args, int* should_exit);
/* -- End -- allocation function*/

/* -- user library in GPU*/
//allocation
__device__ void* allocateThr(size_t allocateSize, struct MMUOnTransfer** pMMUOnTransfer);

//free
__device__ void freeThr(void* freeAddress, struct MMUOnTransfer** pMMUOnTransfer);
/* -- End -- allocation in GPU*/

/*-- End -- Version 1.3.5*/
#endif

