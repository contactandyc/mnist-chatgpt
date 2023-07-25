#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>

typedef struct {
    int size;
    int inputs_size;
    float* biases;
    float** weights;
    float* (*activation_function)(float*, int);
    float* (*activation_derivative)(float*, int);
    float* outputs;
} Layer;

typedef struct {
    int num_layers;
    int input_size; // new field
    Layer** layers;
} NeuralNetwork;

typedef struct {
    int num_samples;
    int input_size;
    int output_size;
    float** inputs;
    float** outputs;
} Dataset;

NeuralNetwork* createNeuralNetwork(int input_size);
Layer* addDenseLayer(NeuralNetwork* network, int size, float* (*activation_function)(float*, int), float* (*activation_derivative)(float*, int));
void train(NeuralNetwork* network, Dataset* data, int epochs, float learning_rate);
void test(NeuralNetwork* network, Dataset* data);

#endif /* NEURAL_NETWORK_H */