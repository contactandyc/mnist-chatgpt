#include "data_processing.h"
#include "ac_allocator.h"
#include <stdlib.h>

// Function to convert pixel data to floating-point values in the range [0, 1]
float* convertToFloat(uint8_t* pixels, size_t num_pixels) {
    // Allocate memory for the output array
    float* floatPixels = (float*)ac_malloc(num_pixels * sizeof(float));
    if (floatPixels == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Convert each pixel value to a floating point value in the range [0, 1]
    for (size_t i = 0; i < num_pixels; i++) {
        float px = pixels[i];
        floatPixels[i] = px / 255.0;
    }

    return floatPixels;
}

// Function to perform one-hot encoding for the input value
float* oneHotEncoding(unsigned char input) {
    // Allocate memory for the output array
    float *output = (float*)ac_malloc(10 * sizeof(float));
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

// Function to process ImageDataWithLabels and create InputAndTargets
InputAndTargets processImageDataWithLabels(ImageDataWithLabels dataWithLabels) {
    InputAndTargets inputAndTargets;

    // Allocate memory for the arrays in InputAndTargets
    inputAndTargets.num_inputs = dataWithLabels.num_images;
    inputAndTargets.inputs = (float **)ac_malloc(inputAndTargets.num_inputs * sizeof(float *));
    if (inputAndTargets.inputs == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    inputAndTargets.num_targets = dataWithLabels.num_images;
    inputAndTargets.targets = (float **)ac_malloc(inputAndTargets.num_targets * sizeof(float *));
    if (inputAndTargets.targets == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Convert images to float
    for (int i = 0; i < inputAndTargets.num_inputs; i++) {
        inputAndTargets.inputs[i] = convertToFloat(dataWithLabels.images[i], dataWithLabels.num_rows * dataWithLabels.num_columns);
    }

    // Convert targets to one-hot encoded floats
    for (int i = 0; i < inputAndTargets.num_targets; i++) {
        inputAndTargets.targets[i] = oneHotEncoding(dataWithLabels.targets[i]);
    }

    return inputAndTargets;
}

// Function to free memory allocated for InputAndTargets structure
void freeInputAndTargets(InputAndTargets *inputAndTargets) {
    for (int i = 0; i < inputAndTargets->num_inputs; i++) {
        ac_free(inputAndTargets->inputs[i]);
    }
    ac_free(inputAndTargets->inputs);

    for (int i = 0; i < inputAndTargets->num_targets; i++) {
        ac_free(inputAndTargets->targets[i]);
    }
    ac_free(inputAndTargets->targets);
}


InputAndTargets loadInputAndTargets(const char *image_filename, const char *label_filename) {
    // Combine image data with label response
    ImageDataWithLabels dataWithLabels = combineImageDataWithLabels(image_filename, label_filename);

    // Allocate memory for InputAndTargets structure
    InputAndTargets inputAndTargets;
    inputAndTargets.num_targets = dataWithLabels.num_targets;
    inputAndTargets.num_inputs = dataWithLabels.num_images;
    inputAndTargets.targets = (float **)ac_malloc(inputAndTargets.num_targets * sizeof(float *));
    inputAndTargets.inputs = (float **)ac_malloc(inputAndTargets.num_inputs * sizeof(float *));
    if (inputAndTargets.targets == NULL || inputAndTargets.inputs == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Convert targets to one-hot encoded floats
    for (int i = 0; i < inputAndTargets.num_targets; i++) {
        inputAndTargets.targets[i] = oneHotEncoding(dataWithLabels.targets[i]);
    }

    // Convert images to float arrays in the range [0, 1]
    for (int i = 0; i < inputAndTargets.num_inputs; i++) {
        inputAndTargets.inputs[i] = convertToFloat(dataWithLabels.images[i], dataWithLabels.num_rows * dataWithLabels.num_columns);
    }

    // Free memory used by ImageDataWithLabels structure
    freeImageDataWithLabels(&dataWithLabels);

    return inputAndTargets;
}
