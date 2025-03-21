#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//generate the random access order and write to the accessOrder.txt
#define ACCESS_NUMBER 60  //the actual access number is ACCESS_NUMBER * 6

int main() {
    int numbers[ACCESS_NUMBER];
    FILE *file;

    srand(time(NULL));

    for (int i = 0; i < ACCESS_NUMBER; i++) {
        numbers[i] = i;
    }

    // shuffle the array
    for (int i = 0; i < ACCESS_NUMBER; i++) {
        int j = rand() % ACCESS_NUMBER;
        int temp = numbers[i];
        numbers[i] = numbers[j];
        numbers[j] = temp;
    }

    // open file
    file = fopen("accessOrder.txt", "w");
    if (file == NULL) {
        printf("The file can not be open\n");
        return 1;
    }

    //write the data
    for (int i = 0; i < ACCESS_NUMBER; i++) {
        fprintf(file, "%d\n", numbers[i]);
    }

    // close the file
    fclose(file);

    return 0;
}