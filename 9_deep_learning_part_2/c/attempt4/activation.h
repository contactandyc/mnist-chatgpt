#ifndef _ACTIVATION_H
#define _ACTIVATION_H

// Sigmoid activation function
float *sigmoid(float* inputs, int size);

// Derivative of the sigmoid function
float *sigmoidDerivative(float* inputs, int size);

// ReLU activation function
float* relu(float* inputs, int size);

// Derivative of ReLU activation function
float* reluDerivative(float* inputs, int size);

// Softmax activation function
float* softmax(float* inputs, int size);

// Derivative of Softmax activation function
// Note: This derivative is not the true mathematical derivative of softmax.
// However, in the context of backpropagation for the output layer with cross-entropy loss,
// this derivative is correct as the loss and softmax derivative get simplified in a way that we only need to
// subtract the target output from the network's output.
float* softmaxDerivative(float* inputs, int size);

#endif /* _ACTIVATION_H */