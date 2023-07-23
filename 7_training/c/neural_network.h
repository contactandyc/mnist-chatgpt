#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "data_processing.h"

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    float **input_to_hidden_weights;
    float **hidden_to_output_weights;
    float *hidden_biases;
    float *output_biases;
} NeuralNetwork;

NeuralNetwork* createNeuralNetwork(int input_size, int hidden_size, int output_size);
void freeNeuralNetwork(NeuralNetwork* model);
float* forwardPass(NeuralNetwork* model, float* input);
void trainNeuralNetwork(NeuralNetwork* model, InputAndTargets* data, int epochs, float learning_rate);
void testModel(NeuralNetwork* model, InputAndTargets* data);

#endif