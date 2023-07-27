#include "initialization.h"

#include "ac_allocator.h"

#include <stdlib.h> // for malloc, calloc and rand
#include <math.h>   // for sqrt
#include <time.h>   // for time

float** zeroInitialization(int num_inputs, int num_outputs) {
    float** weights = (float**)ac_malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++)
        weights[i] = (float*)ac_calloc(num_inputs * sizeof(float));
    return weights;
}

float** randomInitialization(int num_inputs, int num_outputs) {
    // srand((unsigned int)time(NULL)); // Seed for the random number generator

    float** weights = (float**)ac_malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++) {
        weights[i] = (float*)ac_malloc(num_inputs * sizeof(float));
        for (int j = 0; j < num_inputs; j++) {
            weights[i][j] = (float)rand() / RAND_MAX * 2.0 - 1.0; // Generate random float between -1 and 1
        }
    }
    return weights;
}


float** xavierInitialization(int num_inputs, int num_outputs) {
    // srand(time(NULL));
    float** weights = (float**)ac_malloc(num_outputs * sizeof(float*));
    float limit = sqrt(6.0f / (num_inputs + num_outputs));
    for (int i = 0; i < num_outputs; i++) {
        weights[i] = (float*)ac_malloc(num_inputs * sizeof(float));
        for (int j = 0; j < num_inputs; j++)
            weights[i][j] = ((float) rand() / (RAND_MAX)) * 2.0f * limit - limit;
    }
    return weights;
}

float** heInitialization(int num_inputs, int num_outputs) {
    // srand((unsigned int)time(NULL)); // Seed for the random number generator

    float** weights = (float**)ac_malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++) {
        weights[i] = (float*)ac_malloc(num_inputs * sizeof(float));
        float std_dev = sqrt(2.0 / num_inputs);
        for (int j = 0; j < num_inputs; j++) {
            float random_value = (float)rand() / RAND_MAX; // Generate random float between 0 and 1
            weights[i][j] = std_dev * (2.0 * random_value - 1.0); // Generate random float between -std_dev and std_dev
        }
    }
    return weights;
}
