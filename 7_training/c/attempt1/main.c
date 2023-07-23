#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "data_processing.h" // Assuming you've already defined the InputAndTargets structure and related functions

#define INPUT_SIZE 768
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 10

typedef struct {
    int num_targets;
    float **targets;
    int num_inputs;
    float **inputs;
    float *input_layer;
    float *hidden_layer;
    float *output_layer;
    float *hidden_weights;
    float *output_weights;
    float *hidden_biases;
    float *output_biases;
} NeuralNetwork;

// Sigmoid activation function
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
float sigmoid_derivative(float x) {
    float sigmoid_x = sigmoid(x);
    return sigmoid_x * (1.0 - sigmoid_x);
}

// Initialize neural network parameters
void initialize_neural_network(NeuralNetwork *nn) {
    nn->input_layer = (float *)malloc(INPUT_SIZE * sizeof(float));
    nn->hidden_layer = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    nn->output_layer = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    nn->hidden_weights = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    nn->output_weights = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    nn->hidden_biases = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    nn->output_biases = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    // Initialize weights and biases with random values (you can customize this)
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        nn->hidden_weights[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        nn->output_weights[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->hidden_biases[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->output_biases[i] = ((float)rand() / RAND_MAX) - 0.5;
    }
}

// Forward pass through the neural network
void forward_pass(NeuralNetwork *nn) {
    // Calculate hidden layer values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = 0.0;
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += nn->input_layer[j] * nn->hidden_weights[j * HIDDEN_SIZE + i];
        }
        nn->hidden_layer[i] = sigmoid(sum + nn->hidden_biases[i]);
    }

    // Calculate output layer values
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = 0.0;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += nn->hidden_layer[j] * nn->output_weights[j * OUTPUT_SIZE + i];
        }
        nn->output_layer[i] = sigmoid(sum + nn->output_biases[i]);
    }
}

// Backpropagation for updating weights and biases
void backpropagation(NeuralNetwork *nn, float *targets) {
    // Calculate output layer errors
    float output_errors[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_errors[i] = (targets[i] - nn->output_layer[i]) * sigmoid_derivative(nn->output_layer[i]);
    }

    // Calculate hidden layer errors
    float hidden_errors[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float error = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            error += output_errors[j] * nn->output_weights[i * OUTPUT_SIZE + j];
        }
        hidden_errors[i] = error * sigmoid_derivative(nn->hidden_layer[i]);
    }

    // Update output weights and biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->output_weights[j * OUTPUT_SIZE + i] += LEARNING_RATE * output_errors[i] * nn->hidden_layer[j];
        }
        nn->output_biases[i] += LEARNING_RATE * output_errors[i];
    }

    // Update hidden weights and biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->hidden_weights[j * HIDDEN_SIZE + i] += LEARNING_RATE * hidden_errors[i] * nn->input_layer[j];
        }
        nn->hidden_biases[i] += LEARNING_RATE * hidden_errors[i];
    }
}

// Train the neural network using backpropagation
void train_neural_network(NeuralNetwork *nn, InputAndTargets *data) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < data->num_inputs; i++) {
            // Prepare input and targets for the current sample
            for (int j = 0; j < INPUT_SIZE; j++) {
                nn->input_layer[j] = data->inputs[i][j];
            }
            float *targets = data->targets[i];

            // Forward pass
            forward_pass(nn);

            // Backpropagation
            backpropagation(nn, targets);
        }
    }
}

// Clean up memory used by the neural network
void free_neural_network(NeuralNetwork *nn) {
    free(nn->input_layer);
    free(nn->hidden_layer);
    free(nn->output_layer);
    free(nn->hidden_weights);
    free(nn->output_weights);
    free(nn->hidden_biases);
    free(nn->output_biases);
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_filename> <label_filename>\n", argv[0]);
        return 1;
    }

    const char *image_filename = argv[1];
    const char *label_filename = argv[2];

    InputAndTargets data = loadInputAndTargets(image_filename, label_filename);

    // Initialize neural network
    NeuralNetwork nn;
    initialize_neural_network(&nn);

    // Train the neural network
    train_neural_network(&nn, &data);

    // Free memory used by the neural network and data
    free_neural_network(&nn);
    freeInputAndTargets(&data);

    return 0;
}
