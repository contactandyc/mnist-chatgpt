#ifndef _ACTIVATION_H
#define _ACTIVATION_H

// Sigmoid activation function
void sigmoid(float* inputs, int size);

// Derivative of the sigmoid function
void sigmoidDerivative(float* inputs, int size);

// ReLU activation function
void relu(float* inputs, int size);

// Derivative of ReLU activation function
void reluDerivative(float* inputs, int size);

// Softmax activation function
void softmax(float* inputs, int size);

// Derivative of Softmax activation function
// Note: This derivative is not the true mathematical derivative of softmax.
// However, in the context of backpropagation for the output layer with cross-entropy loss,
// this derivative is correct as the loss and softmax derivative get simplified in a way that we only need to
// subtract the target output from the network's output.
void softmaxDerivative(float* inputs, int size);

#endif /* _ACTIVATION_H */