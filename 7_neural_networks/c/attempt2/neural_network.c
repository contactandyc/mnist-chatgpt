#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "neural_network.h"

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

void trainNeuralNetwork(NeuralNetwork* model, InputAndTargets* data, int epochs, float learning_rate) {
    int num_samples = data->num_inputs;
    int num_targets = data->num_targets;

    // Training loop for the specified number of epochs
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
            for (int i = 0; i < num_targets; i++) {
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
            for (int i = 0; i < model->output_size; i++) {
                float output_error = (target[i] - output[i]) * output[i] * (1.0 - output[i]); // Derivative of sigmoid
                for (int j = 0; j < model->hidden_size; j++) {
                    model->hidden_to_output_weights[j][i] += learning_rate * output_error * hidden_layer[j];
                }
                model->output_biases[i] += learning_rate * output_error;
            }

            // Compute hidden layer error and update weights
            for (int i = 0; i < model->hidden_size; i++) {
                float hidden_error = 0.0;
                for (int j = 0; j < model->output_size; j++) {
                    hidden_error += output_error * model->hidden_to_output_weights[i][j];
                }
                hidden_error *= hidden_layer[i] * (1.0 - hidden_layer[i]); // Derivative of sigmoid
                for (int j = 0; j < model->input_size; j++) {
                    model->input_to_hidden_weights[j][i] += learning_rate * hidden_error * input[j];
                }
                model->hidden_biases[i] += learning_rate * hidden_error;
            }

            // Free memory for hidden layer activations as they are no longer needed
            free(hidden_layer);
            free(output);
        }

        // Compute and print the average loss for this epoch
        total_loss /= num_samples;
        printf("Epoch %d - Average Loss: %.6f\n", epoch + 1, total_loss);
    }
}

void testModel(NeuralNetwork* model, InputAndTargets* data) {
    int num_samples = data->num_inputs;
    int num_targets = data->num_targets;

    int correct_predictions = 0;

    for (int sample = 0; sample < num_samples; sample++) {
        float* input = data->inputs[sample];
        float* target = data->targets[sample];

        float* output = forwardPass(model, input);

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

        // Check if the predicted class matches the actual class
        if (predicted_class == actual_class) {
            correct_predictions++;
        }

        free(output);
    }

    float accuracy = (float)correct_predictions / num_samples * 100.0;
    printf("Test Accuracy: %.2f%%\n", accuracy);
}
