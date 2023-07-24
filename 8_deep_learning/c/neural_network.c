#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neural_network.h"


NeuralNetwork* createNeuralNetwork(int input_size, int output_size) {
    // Allocate memory for the neural network structure
    NeuralNetwork* network = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (network == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Initialize the neural network parameters
    network->input_size = input_size;
    network->output_size = output_size;
    network->num_layers = 0;
    network->layers = NULL;  // No layers initially

    return network;
}

float randFloat() {
    return (float)rand() / (float)RAND_MAX;
}

Layer* addDenseLayer(NeuralNetwork* network, int input_size, int size,
                     float* (*activation_function)(float*), float* (*activation_derivative)(float*)) {
    // Create a new layer
    Layer* layer = (Layer*)malloc(sizeof(Layer));

    // Initialize its fields
    layer->input_size = input_size;
    layer->size = size;
    layer->inputs = (float*)malloc(input_size * sizeof(float));
    layer->outputs = (float*)malloc(size * sizeof(float));
    layer->biases = (float*)malloc(size * sizeof(float));
    layer->weights = (float*)malloc(input_size * size * sizeof(float));  // row-major order
    layer->gradients = (float*)malloc(size * sizeof(float));
    layer->activation_function = activation_function;
    layer->activation_derivative = activation_derivative;

    // Initialize biases and weights with small random numbers
    for (int i = 0; i < size; ++i) {
        layer->biases[i] = randFloat();
        for (int j = 0; j < input_size; ++j) {
            layer->weights[i * input_size + j] = randFloat();
        }
    }

    // Add the new layer to the network
    network->layers = (Layer**)realloc(network->layers, (network->num_layers + 1) * sizeof(Layer*));
    network->layers[network->num_layers] = layer;
    network->num_layers++;

    return layer;
}

float* forwardPass(NeuralNetwork* network, float* input) {
    float* layer_input = input;
    float* layer_output;

    // Loop over each layer in the network
    for (int i = 0; i < network->num_layers; i++) {
        Layer* layer = network->layers[i];

        // Allocate memory for the output of this layer
        layer_output = (float*)malloc(sizeof(float) * layer->size);
        if (!layer_output) {
            fprintf(stderr, "Failed to allocate memory for layer output\n");
            exit(1);
        }

        // Perform the weighted sum and activation function for each neuron
        for (int j = 0; j < layer->size; j++) {
            float sum = 0.0;

            // Compute the weighted sum of the inputs
            for (int k = 0; k < layer->input_size; k++) {
                sum += layer->weights[j][k] * layer_input[k];
            }

            // Add the bias and apply the activation function
            layer_output[j] = layer->activation_function(sum + layer->biases[j]);
        }

        // If this isn't the first layer, we need to free the input from the previous layer
        if (i > 0) {
            free(layer_input);
        }

        // The output of this layer will be the input to the next layer
        layer_input = layer_output;
    }

    // The output of the last layer is the output of the network
    return layer_output;
}

float* calculateError(float* output, float* target, int output_size) {
    float* error = malloc(sizeof(float) * output_size);
    for (int i = 0; i < output_size; i++) {
        error[i] = output[i] - target[i];
    }
    return error;
}

void backwardPass(NeuralNetwork* network, float* error, float learning_rate) {
    for (int i = network->num_layers - 1; i >= 0; i--) {
        Layer* layer = network->layers[i];

        // Calculate gradients
        for (int j = 0; j < layer->size; j++) {
            layer->gradients[j] = error[j] * layer->activation_derivative(layer->outputs[j]);
        }

        // Update weights and biases
        for (int j = 0; j < layer->size; j++) {
            for (int k = 0; k < layer->input_size; k++) {
                layer->weights[j * layer->input_size + k] -= learning_rate * layer->gradients[j] * layer->inputs[k];
            }
            layer->biases[j] -= learning_rate * layer->gradients[j];
        }

        // Calculate error for the next layer
        if (i > 0) {
            for (int j = 0; j < layer->input_size; j++) {
                float sum = 0.0;
                for (int k = 0; k < layer->size; k++) {
                    sum += layer->weights[k * layer->input_size + j] * layer->gradients[k];
                }
                error[j] = sum;
            }
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

NeuralNetwork* createNeuralNetwork(int input_size, int hidden_size, int output_size) {
    // Allocate memory for the neural network structure
    NeuralNetwork* model = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (model == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Initialize the neural network parameters
    model->input_size = input_size;
    model->hidden_size = hidden_size;
    model->output_size = output_size;

    // Allocate memory for input-to-hidden weights
    model->input_to_hidden_weights = (float**)malloc(input_size * sizeof(float*));
    if (model->input_to_hidden_weights == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    for (int i = 0; i < input_size; i++) {
        model->input_to_hidden_weights[i] = (float*)malloc(hidden_size * sizeof(float));
        if (model->input_to_hidden_weights[i] == NULL) {
            fprintf(stderr, "Memory allocation error.\n");
            exit(1);
        }
        for (int j = 0; j < hidden_size; j++) {
            // Initialize weights randomly between -0.5 and 0.5
            model->input_to_hidden_weights[i][j] = ((float)rand() / RAND_MAX) - 0.5;
        }
    }

    // Allocate memory for hidden-to-output weights
    model->hidden_to_output_weights = (float**)malloc(hidden_size * sizeof(float*));
    if (model->hidden_to_output_weights == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    for (int i = 0; i < hidden_size; i++) {
        model->hidden_to_output_weights[i] = (float*)malloc(output_size * sizeof(float));
        if (model->hidden_to_output_weights[i] == NULL) {
            fprintf(stderr, "Memory allocation error.\n");
            exit(1);
        }
        for (int j = 0; j < output_size; j++) {
            // Initialize weights randomly between -0.5 and 0.5
            model->hidden_to_output_weights[i][j] = ((float)rand() / RAND_MAX) - 0.5;
        }
    }

    // Allocate memory for hidden biases
    model->hidden_biases = (float*)malloc(hidden_size * sizeof(float));
    if (model->hidden_biases == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    for (int i = 0; i < hidden_size; i++) {
        // Initialize biases randomly between -0.5 and 0.5
        model->hidden_biases[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    // Allocate memory for output biases
    model->output_biases = (float*)malloc(output_size * sizeof(float));
    if (model->output_biases == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }
    for (int i = 0; i < output_size; i++) {
        // Initialize biases randomly between -0.5 and 0.5
        model->output_biases[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    return model;
}

float* forwardPass(NeuralNetwork* model, float* input) {
    // Allocate memory for the hidden layer
    float* hidden_layer = (float*)malloc(model->hidden_size * sizeof(float));
    if (hidden_layer == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Compute hidden layer activations
    for (int i = 0; i < model->hidden_size; i++) {
        hidden_layer[i] = 0.0;
        for (int j = 0; j < model->input_size; j++) {
            hidden_layer[i] += input[j] * model->input_to_hidden_weights[j][i];
        }
        hidden_layer[i] += model->hidden_biases[i];
        hidden_layer[i] = 1.0 / (1.0 + expf(-hidden_layer[i])); // Apply sigmoid activation
    }

    // Compute output layer activations
    float* output = (float*)malloc(model->output_size * sizeof(float));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    for (int i = 0; i < model->output_size; i++) {
        output[i] = 0.0;
        for (int j = 0; j < model->hidden_size; j++) {
            output[i] += hidden_layer[j] * model->hidden_to_output_weights[j][i];
        }
        output[i] += model->output_biases[i];
        output[i] = 1.0 / (1.0 + expf(-output[i])); // Apply sigmoid activation
    }

    // Free memory for hidden layer as it's no longer needed
    free(hidden_layer);

    return output;
}

void freeNeuralNetwork(NeuralNetwork* model) {
    for (int i = 0; i < model->input_size; i++) {
        free(model->input_to_hidden_weights[i]);
    }
    for (int i = 0; i < model->hidden_size; i++) {
        free(model->hidden_to_output_weights[i]);
    }
    free(model->input_to_hidden_weights);
    free(model->hidden_to_output_weights);
    free(model->hidden_biases);
    free(model->output_biases);
    free(model);
}

void print_output(const char *title, float *o, int num) {
    printf("%s: [", title);
    for (int i = 0; i < num; i++) {
        printf("%.4f ", o[i]);
    }
    printf("]\n");
}

void trainNeuralNetwork(NeuralNetwork* model, InputAndTargets* data, int epochs, float learning_rate) {
    int num_samples = data->num_inputs;

    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;

        // Loop through each input data sample
        for (int sample = 0; sample < num_samples; sample++) {
            float* input = data->inputs[sample];
            float* target = data->targets[sample];

            // Forward pass
            float* output = forwardPass(model, input);

            // Compute loss (mean squared error)
            float loss = 0.0;
            for (int i = 0; i < model->output_size; i++) {
                loss += (target[i] - output[i]) * (target[i] - output[i]);
            }
            total_loss += loss;

            // Backpropagation and weight update
            float* hidden_layer = (float*)malloc(model->hidden_size * sizeof(float));
            if (hidden_layer == NULL) {
                fprintf(stderr, "Memory allocation error.\n");
                exit(1);
            }

            // Compute hidden layer activations
            for (int i = 0; i < model->hidden_size; i++) {
                hidden_layer[i] = 0.0;
                for (int j = 0; j < model->input_size; j++) {
                    hidden_layer[i] += input[j] * model->input_to_hidden_weights[j][i];
                }
                hidden_layer[i] += model->hidden_biases[i];
                hidden_layer[i] = 1.0 / (1.0 + expf(-hidden_layer[i])); // Apply sigmoid activation
            }

            // Compute output layer error and update weights
            float* output_error = (float*)malloc(model->output_size * sizeof(float));
            if (output_error == NULL) {
                fprintf(stderr, "Memory allocation error.\n");
                exit(1);
            }
            for (int i = 0; i < model->output_size; i++) {
                output_error[i] = (target[i] - output[i]) * output[i] * (1.0 - output[i]); // Derivative of sigmoid
                for (int j = 0; j < model->hidden_size; j++) {
                    model->hidden_to_output_weights[j][i] += learning_rate * output_error[i] * hidden_layer[j];
                }
                model->output_biases[i] += learning_rate * output_error[i];
            }

            // Compute hidden layer error and update weights
            for (int i = 0; i < model->hidden_size; i++) {
                float hidden_error = 0.0;
                for (int j = 0; j < model->output_size; j++) {
                    hidden_error += output_error[j] * model->hidden_to_output_weights[i][j];
                }
                hidden_error *= hidden_layer[i] * (1.0 - hidden_layer[i]); // Derivative of sigmoid
                for (int j = 0; j < model->input_size; j++) {
                    model->input_to_hidden_weights[j][i] += learning_rate * hidden_error * input[j];
                }
                model->hidden_biases[i] += learning_rate * hidden_error;
            }

            // Free memory for hidden layer activations and output error as they are no longer needed
            free(hidden_layer);
            free(output_error);
            free(output);
        }

        // Compute and print the average loss for this epoch
        total_loss /= num_samples;
        printf("Epoch %d - Average Loss: %.6f\n", epoch + 1, total_loss);
    }
}


void testModel(NeuralNetwork* model, InputAndTargets* data) {
    int num_samples = data->num_inputs;
    int correct_predictions = 0;

    for (int sample = 0; sample < num_samples; sample++) {
        float* input = data->inputs[sample];
        float* target = data->targets[sample];

        float* output = forwardPass(model, input);

        // Find the index of the maximum value in the output (predicted class)
        int predicted_class = 0;
        for (int i = 1; i < model->output_size; i++) {
            if (output[i] > output[predicted_class]) {
                predicted_class = i;
            }
        }

        // Find the index of the maximum value in the target (actual class)
        int actual_class = 0;
        for (int i = 1; i < model->output_size; i++) {
            if (target[i] > target[actual_class]) {
                actual_class = i;
            }
        }

        // Check if the predicted class matches the actual class
        if (predicted_class == actual_class) {
            correct_predictions++;
        }

        free(output);
    }

    float accuracy = (float)correct_predictions / num_samples * 100.0;
    printf("Test Accuracy: %.2f%%\n", accuracy);
}
