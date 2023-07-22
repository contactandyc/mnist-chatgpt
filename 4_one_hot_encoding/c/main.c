#include <stdio.h>
#include "one_hot_encoding.h"

int main() {
    unsigned char input = 5;
    float *output = oneHotEncoding(input);

    // Print the one-hot encoded array
    printf("Input %hhu: [", input);
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", output[i]);
    }
    printf("]\n");

    // Free the memory allocated for the one-hot encoded array
    freeOneHotEncoded(output);

    return 0;
}