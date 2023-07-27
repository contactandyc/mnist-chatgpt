#include "neural_network.h"
#include "matrix.h"
#include "ac_allocator.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


NeuralNetwork* createNeuralNetwork(int input_size) {
    NeuralNetwork* network = ac_malloc(sizeof(NeuralNetwork));
    network->num_layers = 0;
    network->input_size = input_size; // set input_size
    network->layers = NULL;
    return network;
}

void freeNetworkLayer(Layer *layer) {
    ac_free(layer->biases);
    freeMatrix(layer->weights, layer->size);
    ac_free(layer->outputs);
    ac_free(layer->output_error);
    freeMatrix(layer->gradients, layer->size);
    freeMatrix(layer->transposed, layer->inputs_size);
    ac_free(layer);
}

void freeNeuralNetwork(NeuralNetwork *network) {
    if(network->layers) {
        for( int i=0; i<network->num_layers; i++ )
            freeNetworkLayer(network->layers[i]);
        free(network->layers);
    }
    ac_free(network);
}

float* createRandomArray(int size) {
    // srand(time(NULL)); // seed the random number generator with the current time
    float* array = ac_malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        array[i] = ((float) rand() / (RAND_MAX)) - 0.5f; // generate a random float between -0.5 and 0.5
    }
    return array;
}

float** createRandomMatrix(int rows, int cols) {
    // srand(time(NULL)); // seed the random number generator with the current time
    float** matrix = ac_malloc(sizeof(float*) * rows);
    for (int i = 0; i < rows; i++) {
        matrix[i] = ac_malloc(sizeof(float) * cols);
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = ((float) rand() / (RAND_MAX)) - 0.5f; // generate a random float between -0.5 and 0.5
        }
    }
    return matrix;
}

// #define DEBUG_NETWORK
#ifdef DEBUG_NETWORK
void printVector(float *o, int num) {
    printf("[");
    for (int i = 0; i < num; i++) {
        printf("%.4f ", o[i]);
    }
    printf("]\n");
}

void testVector(float* o, int num) {
    for( int i=0; i<num; i++ ) {
        if(isnan(o[i]) || isinf(o[i])) {
            printVector(o, num);
            abort();
        }
    }
}

void testMatrix(float** m, int a, int b) {
    for( int i=0; i<a; i++ )
        testVector(m[i], b);
}

void testNetwork(NeuralNetwork* network) {
    for( int i=0; i<network->num_layers; i++ ) {
        Layer *layer = network->layers[i];
        testVector(layer->biases, layer->size);
        testMatrix(layer->weights, layer->size, layer->inputs_size);
    }
}
#else
#define printVector(a, b) ;
#define testVector(a, b) ;
#define testMatrix(a, b, c) ;
#define testNetwork(a) ;
#endif

Layer* addDenseLayer(NeuralNetwork* network, int size, void (*activation_function)(float*, int), void (*activation_derivative)(float*, int)) {
    Layer* layer = (Layer*)ac_malloc(sizeof(Layer));
    int inputs_size;
    if (network->num_layers == 0) {
        inputs_size = network->input_size;
    } else {
        inputs_size = network->layers[network->num_layers-1]->size;
    }

    // Initialize the layer
    layer->size = size;
    layer->inputs_size = inputs_size;
    layer->biases = createRandomArray(size);
    layer->weights = createRandomMatrix(size, inputs_size);
    layer->activation_function = activation_function;
    layer->activation_derivative = activation_derivative;
    layer->outputs = (float*)ac_calloc(size * sizeof(float));
    layer->output_error = (float*)ac_calloc(inputs_size * sizeof(float));
    layer->gradients = createZeroMatrix(size, inputs_size);
    layer->transposed = createZeroMatrix(inputs_size, size);

    // Reallocate memory for the layers array
    network->layers = realloc(network->layers, sizeof(Layer*) * (network->num_layers + 1));

    // Add the new layer to the network
    network->layers[network->num_layers] = layer;
    network->num_layers++;

    return layer;
}

float* forwardPass(NeuralNetwork* network, float* inputs) {
    float* layer_inputs = inputs;
    for (int i = 0; i < network->num_layers; i++) {
        Layer* layer = network->layers[i];
        matrixMultiply(layer->weights, layer_inputs, layer->outputs, layer->size, layer->inputs_size);
        vectorAdd(layer->outputs, layer->biases, layer->size);
        layer->activation_function(layer->outputs, layer->size);
        layer_inputs = layer->outputs;
    }
    return layer_inputs;
}

float* calculateError(float* output, float* target, float *error, int size) {
    // float* error = (float*)ac_malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        error[i] = output[i] - target[i];
    }
    return error;
}

void backwardPass(NeuralNetwork* network, float *input, float* error, float learning_rate) {
    // Compute the error derivative of the output layer
    float* output_error = error;
    for (int i = network->num_layers-1; i >= 0; i--) {
        Layer* layer = network->layers[i];
        // float* layer_inputs = i == 0 ? output_layer->outputs : network->layers[i-1]->outputs;  // modified line
        float* layer_inputs = i == 0 ? input : network->layers[i-1]->outputs;  // modified line

        testNetwork(network);
        testVector(layer_inputs, layer->inputs_size);
        testVector(output_error, layer->size);

        // Compute the derivative of the error with respect to weights and biases
        float** weight_gradients = outerProduct(output_error, layer_inputs, layer->gradients, layer->size, layer->inputs_size);
        float* bias_gradients = output_error;

        testMatrix(weight_gradients, layer->size, layer->inputs_size);

        // Update weights and biases
        matrixSubtract(layer->weights, weight_gradients, layer->size, layer->inputs_size, learning_rate);

        testNetwork(network);

        vectorSubtractWithLearningRate(layer->biases, bias_gradients, layer->size, learning_rate);

        testNetwork(network);

        if (i != 0) {
            // Compute the error derivative for the next lower layer
            float **transposed = transpose(layer->weights, layer->transposed, layer->size, layer->inputs_size);
            testMatrix(transposed, layer->inputs_size, layer->size);
            testVector(output_error, layer->inputs_size);

            output_error = matrixVectorMultiply(transposed, output_error, layer->output_error, layer->inputs_size, layer->size);
            testVector(output_error, layer->inputs_size);

            layer->activation_derivative(layer_inputs, layer->inputs_size);
            testVector(layer_inputs, layer->inputs_size);

            output_error = elementwiseMultiply(output_error, layer_inputs, layer->inputs_size);
            testVector(output_error, layer->inputs_size);
        }
    }
}

void train(NeuralNetwork* network, Dataset* data, int epochs, float learning_rate) {
    float *output_error = network->layers[network->num_layers - 1]->output_error;
    int output_size = network->layers[network->num_layers - 1]->size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        for (int i = 0; i < data->num_samples; i++) {
            float* input = data->inputs[i];
            float* target = data->outputs[i];

            // Forward pass: compute the output of the network
            float* output = forwardPass(network, input);

            // Compute the error of the output
            float* error = calculateError(output, target, output_error, output_size);

            for (int j = 0; j < output_size; j++) {
                total_loss += error[j] * error[j]; // MSE loss
            }

            testNetwork(network);

            // Backward pass: adjust the weights and biases of the network based on the error
            backwardPass(network, input, error, learning_rate);

            testNetwork(network);
        }

        total_loss /= data->num_samples;
        printf("Epoch %d: loss = %.5f\n", epoch + 1, total_loss);
    }
}

void test(NeuralNetwork* network, Dataset* data) {
    int num_samples = data->num_samples;
    int num_targets = network->layers[network->num_layers - 1]->size;

    int correct_predictions = 0;

    // Loop over each sample in the dataset
    for (int sample = 0; sample < num_samples; sample++) {
        float* input = data->inputs[sample];
        float* target = data->outputs[sample];

        // Perform a forward pass through the network
        float* output = forwardPass(network, input);

        // Find the index of the maximum value in the output (predicted class)
        int predicted_class = 0;
        for (int i = 1; i < num_targets; i++) {
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

        // free(output);
    }

    // Calculate and print the accuracy
    float accuracy = (float)correct_predictions / num_samples * 100.0;
    printf("Test Accuracy: %.2f%%\n", accuracy);
}