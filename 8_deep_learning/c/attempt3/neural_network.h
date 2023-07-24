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
    Layer* layers;
} NeuralNetwork;

typedef struct {
    int num_samples;
    int input_size;
    int output_size;
    float** inputs;
    float** outputs;
} Dataset;

float randFloat();

NeuralNetwork* createNeuralNetwork();
Layer* addDenseLayer(NeuralNetwork* network, int size, float* (*activation_function)(float*, int), float* (*activation_derivative)(float*, int));
void forwardPass(NeuralNetwork* network, float* input);
float calculateError(NeuralNetwork* network, float* target);
void backwardPass(NeuralNetwork* network, float* target, float learning_rate);
void train(NeuralNetwork* network, Dataset* data, int epochs, float learning_rate);
void test(NeuralNetwork* network, Dataset* data);

#endif /* NEURAL_NETWORK_H */