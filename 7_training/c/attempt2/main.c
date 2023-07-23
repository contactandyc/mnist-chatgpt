#include <stdio.h>
#include <stdlib.h>
#include "data_processing.h"
#include "neural_network.h"

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_filename> <label_filename>\n", argv[0]);
        return 1;
    }

    const char *image_filename = argv[1];
    const char *label_filename = argv[2];

    // Load input and target data
    InputAndTargets data = loadInputAndTargets(image_filename, label_filename);

    // Define neural network hyperparameters
    int input_size = 768; // Input size (number of features)
    int hidden_size = 128; // Number of hidden units
    int output_size = 10; // Number of output classes
    int epochs = 10; // Number of training epochs
    float learning_rate = 0.01;

    // Create and train the neural network
    NeuralNetwork* model = createNeuralNetwork(input_size, hidden_size, output_size);
    trainNeuralNetwork(model, &data, epochs, learning_rate);

    // Test the trained model
    testModel(model, &data);

    // Clean up memory
    freeNeuralNetwork(model);
    freeInputAndTargets(&data);

    return 0;
}
