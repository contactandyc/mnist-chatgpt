#include "activation.h"

#include <stdlib.h>
#include <math.h>

// Sigmoid activation function
float* sigmoid(float* inputs, int size) {
    float* outputs = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        outputs[i] = 1.0f / (1.0f + exp(-inputs[i]));
    }
    return outputs;
}

// Derivative of the sigmoid function
float* sigmoidDerivative(float* inputs, int size) {
    float* outputs = (float*)malloc(size * sizeof(float));
    for( int i = 0; i < size; i++ ) {
        float sigmoid_x = 1.0f / (1.0f + exp(-inputs[i]));
        outputs[i] = sigmoid_x * (1.0f - sigmoid_x);
    }
    return outputs;
}

// ReLU activation function
float* relu(float* inputs, int size) {
    float* outputs = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
    }
    return outputs;
}

// Derivative of ReLU activation function
float* reluDerivative(float* inputs, int size) {
    float* outputs = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        outputs[i] = inputs[i] > 0 ? 1.0f : 0.0f;
    }
    return outputs;
}

// Softmax activation function
float* softmax(float* inputs, int size) {
    float* outputs = (float*)malloc(size * sizeof(float));
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
        outputs[i] = exp(inputs[i] - max);
        sum += outputs[i];
    }

    for (int i = 0; i < size; i++) {
        outputs[i] /= sum;
    }

    return outputs;
}

// Derivative of Softmax activation function
// Note: This derivative is not the true mathematical derivative of softmax.
// However, in the context of backpropagation for the output layer with cross-entropy loss,
// this derivative is correct as the loss and softmax derivative get simplified in a way that we only need to
// subtract the target output from the network's output.
float* softmaxDerivative(float* inputs, int size) {
    float* outputs = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        outputs[i] = inputs[i];
    }
    return outputs;
}
