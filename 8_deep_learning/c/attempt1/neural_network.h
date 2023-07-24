#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdlib.h>

typedef struct {
    int size;
    float* inputs;
    float* outputs;
    float* biases;
    float* weights;
    float* gradients;
    float* (*activation_function)(float*);
    float* (*activation_derivative)(float*);
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

NeuralNetwork* createNeuralNetwork(int input_size, int output_size);
Layer* addDenseLayer(NeuralNetwork* network, int input_size, int size, float* (*activation_function)(float*), float* (*activation_derivative)(float*));
void forwardPass(NeuralNetwork* network, float* input);
float calculateError(NeuralNetwork* network, float* target);
void backwardPass(NeuralNetwork* network, float* target, float learning_rate);
void train(NeuralNetwork* network, Dataset* data, int epochs, float learning_rate);
void test(NeuralNetwork* network, Dataset* data);

#endif /* NEURAL_NETWORK_H */