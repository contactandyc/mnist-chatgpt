#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "neural_network.h"
#include "matrix.h"

NeuralNetwork* createNeuralNetwork() {
    NeuralNetwork* network = malloc(sizeof(NeuralNetwork));
    network->num_layers = 0;
    network->layers = NULL;
    return network;
}

float* createRandomArray(int size) {
    srand(time(NULL)); // seed the random number generator with the current time
    float* array = malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        array[i] = ((float) rand() / (RAND_MAX)) - 0.5f; // generate a random float between -0.5 and 0.5
    }
    return array;
}

float** createRandomMatrix(int rows, int cols) {
    srand(time(NULL)); // seed the random number generator with the current time
    float** matrix = malloc(sizeof(float*) * rows);
    for (int i = 0; i < rows; i++) {
        matrix[i] = malloc(sizeof(float) * cols);
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = ((float) rand() / (RAND_MAX)) - 0.5f; // generate a random float between -0.5 and 0.5
        }
    }
    return matrix;
}

Layer* addDenseLayer(NeuralNetwork* network, int size, float* (*activation_function)(float*, int), float* (*activation_derivative)(float*, int)) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    int inputs_size = network->layers[network->num_layers-1]->size;
    layer->size = size;
    layer->inputs_size = inputs_size;
    layer->biases = createRandomArray(size);
    layer->weights = createRandomMatrix(size, inputs_size);
    layer->activation_function = activation_function;
    layer->activation_derivative = activation_derivative;
    layer->outputs = (float*)calloc(size, sizeof(float));

    network->layers[network->num_layers] = layer;
    network->num_layers++;
    return layer;
}

float* forwardPass(NeuralNetwork* network, float* inputs) {
    for (int i = 0; i < network->num_layers; i++) {
        Layer* layer = network->layers[i];
        float* layer_inputs = i == 0 ? inputs : network->layers[i-1]->outputs;
        matrixMultiply(layer->weights, layer_inputs, layer->outputs, layer->size, layer->inputs_size);
        vectorAdd(layer->outputs, layer->biases, layer->size);
        layer->outputs = layer->activation_function(layer->outputs, layer->size);
    }
    return network->layers[network->num_layers-1]->outputs;
}

float calculateError(NeuralNetwork* network, float* target) {
    Layer* output_layer = network->layers[network->num_layers-1];
    float* output_error = vectorSubtract(output_layer->outputs, target, output_layer->size);
    return 0.5f * dotProduct(output_error, output_error, output_layer->size);
}

void backwardPass(NeuralNetwork* network, float* target, float learning_rate) {
    Layer* output_layer = network->layers[network->num_layers-1];
    // Compute the error derivative of the output layer
    float* output_error = vectorSubtract(output_layer->outputs, target, output_layer->size);
    for (int i = network->num_layers-1; i >= 0; i--) {
        Layer* layer = network->layers[i];
        float* layer_inputs = i == 0 ? output_layer->outputs : network->layers[i-1]->outputs;  // modified line

        // Compute the derivative of the error with respect to weights and biases
        float** weight_gradients = outerProduct(output_error, layer_inputs, layer->size, layer->inputs_size);
        float* bias_gradients = output_error;

        // Update weights and biases
        matrixSubtract(layer->weights, weight_gradients, layer->size, layer->inputs_size, learning_rate);
        vectorSubtractWithLearningRate(layer->biases, bias_gradients, layer->size, learning_rate);

        if (i != 0) {
            // Compute the error derivative for the next lower layer
            output_error = matrixVectorMultiply(transpose(layer->weights, layer->size, layer->inputs_size), output_error, layer->inputs_size, layer->size);
            output_error = elementwiseMultiply(output_error, layer->activation_derivative(layer_inputs, layer->inputs_size), layer->inputs_size);
        }
    }
}

void train(NeuralNetwork* network, Dataset* data, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        for (int i = 0; i < data->num_examples; i++) {
            float* input = data->inputs[i];
            float* target = data->targets[i];

            // Forward pass: compute the output of the network
            float* output = forwardPass(network, input);

            // Compute the error of the output
            float* error = calculateError(output, target, network->output_size);

            // Backward pass: adjust the weights and biases of the network based on the error
            backwardPass(network, error, learning_rate);

            // Compute the total loss for this epoch
            for (int j = 0; j < network->output_size; j++) {
                total_loss += error[j] * error[j]; // MSE loss
            }

            // Clean up
            free(output);
            free(error);
        }

        total_loss /= data->num_examples;
        printf("Epoch %d: loss = %.5f\n", epoch + 1, total_loss);
    }
}

void test(NeuralNetwork* network, Dataset* data) {
    int num_samples = data->num_inputs;
    int num_targets = data->num_targets;

    int correct_predictions = 0;

    // Loop over each sample in the dataset
    for (int sample = 0; sample < num_samples; sample++) {
        float* input = data->inputs[sample];
        float* target = data->targets[sample];

        // Perform a forward pass through the network
        float* output = forwardPass(network, input);

        // Find the index of the maximum value in the output (predicted class)
        int predicted_class = 0;
        for (int i = 1; i < network->output_size; i++) {
            if (output[i] > output[predicted_class]) {
                predicted_class = i;
            }
        }

        // Find the index of the maximum value in the target (actual class)
        int actual_class = 0;
        for (int i = 1; i < num_targets; i++) {
            if (target[i] > target[actual_class]) {
                actual_class = i;
            }
        }

        // If the predicted class matches the actual class, increment the counter
        if (predicted_class == actual_class) {
            correct_predictions++;
        }

        free(output);
    }

    // Calculate and print the accuracy
    float accuracy = (float)correct_predictions / num_samples * 100.0;
    printf("Test Accuracy: %.2f%%\n", accuracy);
}
