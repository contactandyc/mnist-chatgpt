#include <stdio.h>
#include "image_utils.h"
#include "data_processing.h"

int main(int argc, char *argv[]) {
    // Load the ImageData and LabelResponse files
    ImageDataWithLabels dataWithLabels = combineImageDataWithLabels(argv[1], argv[2]);

    // Process the data and create InputAndTargets structure
    InputAndTargets inputAndTargets = processImageDataWithLabels(dataWithLabels);

    // Print some information about the data
    printf("Number of images: %d\n", inputAndTargets.num_inputs);
    printf("Number of targets: %d\n", inputAndTargets.num_targets);

    // Print the first 10 targets for demonstration
    printf("First 10 targets: [");
    for (int i = 0; i < 10; i++) {
        printf("%f ", inputAndTargets.targets[0][i]);
    }
    printf("]\n");

    // Print the first 10 inputs for demonstration
    printf("First 10 inputs: [");
    for (int i = 0; i < 10; i++) {
        printf("%f ", inputAndTargets.inputs[0][i]);
    }
    printf("]\n");

    // Free the memory allocated for ImageDataWithLabels and InputAndTargets structures
    freeImageDataWithLabels(&dataWithLabels);
    freeInputAndTargets(&inputAndTargets);

    return 0;
}