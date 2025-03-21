#include <stdio.h>
#include <stdbool.h>

#define NUM_LEAVES 16
#define TOTAL_NODES (2*NUM_LEAVES - 1)

bool tree[TOTAL_NODES];

int parent(int idx) {
    // if (idx == 0) return -1;
    return (idx - 1) / 2;
}

int leftChild(int idx) {
    return 2 * idx + 1;
}

int rightChild(int idx) {
    return 2 * idx + 2;
}

//a middle function to update the node for recursion
void updateTree(int idx, bool value) {
    // Start from the given index, update the value and propagate up
    tree[idx] = value;
    while (idx > 0) {
        int p = parent(idx);
        tree[p] = tree[leftChild(p)] | tree[rightChild(p)];
        idx = p;
    }
}

//set the leaf index to 1 and update the tree nodes
void insert(int leaf_idx) {
    // Convert leaf index to the corresponding index in the array
    int node_idx = NUM_LEAVES - 1 + leaf_idx;
    updateTree(node_idx, true);
}

//set the leaf index to 0 and update the tree nodes
void delete(int leaf_idx) {
    int node_idx = NUM_LEAVES - 1 + leaf_idx;
    updateTree(node_idx, false);
}

void printTree() {
    for (int i = 0; i < TOTAL_NODES; i++) {
        printf("%d: %d\n", i, tree[i]);
    }
}

//find the 1 in first fit strategy
int findMinOne() {
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
int findSuccessor(int leaf_idx) {
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
int findFirstFitKConsecutiveOnes(int k) {
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

void setZeroConsecutiveKOnesOptimized(int startIdx, int k) {
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

void setOneConsecutiveKZeros(int startIdx, int k) {
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

int main() {
    // Initialize all nodes to 0
    for (int i = 0; i < TOTAL_NODES; i++) {
        tree[i] = false;
    }

    // // Example usage
    // insert(5);
    // insert(3);
    // insert(1);
    // printf("Index of the smallest 1: %d\n", findMinOne());
    
    // delete(1);
    // printTree();
    // printf("Index of the smallest 1 after deleting index 3: %d\n", findMinOne());

    // Example usage
    insert(10);
    insert(15);
    printTree();
    printf("The successor of leaf 5 is: %d\n", findSuccessor(15));

    return 0;
}