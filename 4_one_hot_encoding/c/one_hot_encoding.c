#include <stdio.h>
#include <stdlib.h>

float* oneHotEncoding(unsigned char input) {
    // Allocate memory for the output array
    float *output = (float*)malloc(10 * sizeof(float));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Initialize the output array to all zeros
    for (int i = 0; i < 10; i++) {
        output[i] = 0.0;
    }

    // Perform one-hot encoding for the input value
    if (input < 10) {
        output[input] = 1.0;
    } else {
        fprintf(stderr, "Invalid input value: %hhu\n", input);
        exit(1);
    }

    return output;
}

void freeOneHotEncoded(float* encoded) {
    free(encoded);
}