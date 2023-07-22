#include <stdio.h>
#include <stdlib.h>

float* convertToFloat(uint8_t* pixels, size_t num_pixels) {
    // Allocate memory for the output array
    float* floatPixels = (float*)malloc(num_pixels * sizeof(float));
    if (floatPixels == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Convert each pixel value to a floating point value in the range [0, 1]
    for (size_t i = 0; i < num_pixels; i++) {
        floatPixels[i] = (float)pixels[i] / 255.0;
    }

    return floatPixels;
}

void printFloatPixels(float* floatPixels, size_t num_pixels) {
    for (size_t i = 0; i < num_pixels; i++) {
        printf("%.2f ", floatPixels[i]);
    }
    printf("\n");
}

int main() {
    uint8_t pixels[] = {0, 127, 255};
    size_t num_pixels = sizeof(pixels) / sizeof(pixels[0]);

    float* floatPixels = convertToFloat(pixels, num_pixels);

    // Print the converted floating point pixel values
    printFloatPixels(floatPixels, num_pixels);

    // Free the memory allocated for the floatPixels array
    free(floatPixels);

    return 0;
}