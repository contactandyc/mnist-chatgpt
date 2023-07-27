#ifndef _INITIALIZATION_H
#define _INITIALIZATION_H

// Function prototypes for various weight initialization strategies.

// zeroInitialization: This function initializes a 2D array of weights
// of specified size with all elements being zeros.
//
// @param num_inputs: The number of input neurons to the layer for which weights are being initialized.
// @param num_outputs: The number of neurons in the layer for which weights are being initialized.
//
// @return: A 2D array of weights initialized to zeros.
float** zeroInitialization(int num_inputs, int num_outputs);

// randomInitialization: This function initializes a 2D array of weights
// of specified size with random values between -1 and 1.
//
// @param num_inputs: The number of input neurons to the layer for which weights are being initialized.
// @param num_outputs: The number of neurons in the layer for which weights are being initialized.
//
// @return: A 2D array of weights initialized to random values.
float** randomInitialization(int num_inputs, int num_outputs);

// xavierInitialization: This function initializes a 2D array of weights
// of specified size using the Xavier/Glorot initialization method. It's specifically designed for layers
// with sigmoid or tanh activation functions.
//
// @param num_inputs: The number of input neurons to the layer for which weights are being initialized.
// @param num_outputs: The number of neurons in the layer for which weights are being initialized.
//
// @return: A 2D array of weights initialized using Xavier/Glorot initialization.
float** xavierInitialization(int num_inputs, int num_outputs);

// heInitialization: This function initializes a 2D array of weights
// of specified size using the He initialization method. It's specifically designed for layers
// with ReLU or variants of ReLU activation functions.
//
// @param num_inputs: The number of input neurons to the layer for which weights are being initialized.
// @param num_outputs: The number of neurons in the layer for which weights are being initialized.
//
// @return: A 2D array of weights initialized using He initialization.
float** heInitialization(int num_inputs, int num_outputs);

#endif /* _INITIALIZATION_H */