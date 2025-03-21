#include <stdio.h>

#define GROUP_SIZE 4
#define TOTAL_ELEMENTS 16

void printGroups(int arr[], int start, int length) {
    int startGroup = start / GROUP_SIZE;
    int startOffset = start % GROUP_SIZE;
    int endGroup = (start + length - 1) / GROUP_SIZE;
    int endOffset = (start + length - 1) % GROUP_SIZE;

    for (int i = startGroup; i <= endGroup; i++) {
        int startOffsetCopy = 0;
        int endOffsetCopy = GROUP_SIZE - 1;
        if(i == startGroup)
        {
            startOffsetCopy = startOffset;
        }
        if(i == endGroup)
        {
            endOffsetCopy = endOffset;
        }

        for(int j = startOffsetCopy; j <= endOffsetCopy; j++)
        {
            printf("%d ", arr[j]);
        }
        printf("\n");
    }
}

int main() {
    int arr[TOTAL_ELEMENTS] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    int startIndex = 5;
    int length = 7;

    printGroups(arr, startIndex, length);

    return 0;
}