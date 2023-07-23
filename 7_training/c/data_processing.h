#ifndef DATA_PROCESSING_H
#define DATA_PROCESSING_H

#include <stdint.h>
#include <stdio.h>
#include "image_utils.h"

typedef struct {
    int num_targets;
    float **targets;
    int num_inputs;
    float **inputs;
} InputAndTargets;

// Function to convert pixel data to floating-point values in the range [0, 1]
float* convertToFloat(uint8_t* pixels, size_t num_pixels);

// Function to perform one-hot encoding for the input value
float* oneHotEncoding(unsigned char input);

// Function to process ImageDataWithLabels and create InputAndTargets
InputAndTargets processImageDataWithLabels(ImageDataWithLabels dataWithLabels);

// Function to free memory allocated for InputAndTargets structure
void freeInputAndTargets(InputAndTargets *inputAndTargets);

InputAndTargets loadInputAndTargets(const char *image_filename, const char *label_filename);

#endif /* DATA_PROCESSING_H */