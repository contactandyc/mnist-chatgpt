#include "activation.h"

#include <stdlib.h>
#include <math.h>

// Sigmoid activation function
void sigmoid(float* inputs, int size) {
    for (int i = 0; i < size; i++) {
        inputs[i] = 1.0f / (1.0f + exp(-inputs[i]));
    }
}

// Derivative of the sigmoid function
void sigmoidDerivative(float* inputs, int size) {
    for( int i = 0; i < size; i++ ) {
        float sigmoid_x = 1.0f / (1.0f + exp(-inputs[i]));
        inputs[i] = sigmoid_x * (1.0f - sigmoid_x);
    }
}

// ReLU activation function
void relu(float* inputs, int size) {
    for (int i = 0; i < size; i++) {
        inputs[i] = inputs[i] > 0 ? inputs[i] : 0.001;
    }
}

// Derivative of ReLU activation function
void reluDerivative(float* inputs, int size) {
    for (int i = 0; i < size; i++) {
        inputs[i] = inputs[i] > 0 ? 0.999f : 0.001f;
    }
}

// Softmax activation function
void softmax(float* inputs, int size) {
    float max = inputs[0];
    float sum = 0.0f;

    // Find maximum element to avoid overflow during exponentiation
    for (int i = 0; i < size; i++) {
        if (inputs[i] > max) {
            max = inputs[i];
        }
    }

    // Compute softmax values
    for (int i = 0; i < size; i++) {
        inputs[i] = exp(inputs[i] - max);
        sum += inputs[i];
    }

    for (int i = 0; i < size; i++) {
        inputs[i] /= sum;
    }
}

// Derivative of Softmax activation function
// Note: This derivative is not the true mathematical derivative of softmax.
// However, in the context of backpropagation for the output layer with cross-entropy loss,
// this derivative is correct as the loss and softmax derivative get simplified in a way that we only need to
// subtract the target output from the network's output.
void softmaxDerivative(float* inputs, int size) {
    (void)(inputs);
    (void)(size);
    // do nothing at all
}
