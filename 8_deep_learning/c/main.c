#include <stdio.h>
#include <stdlib.h>
#include "data_processing.h"
#include "neural_network.h"

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <train_image_filename> <train_label_filename> <test_image_filename> <test_label_filename>\n", argv[0]);
        return 1;
    }

    const char *train_image_filename = argv[1];
    const char *train_label_filename = argv[2];
    const char *test_image_filename = argv[3];
    const char *test_label_filename = argv[4];

    // Load input and target data
    InputAndTargets data = loadInputAndTargets(train_image_filename, train_label_filename);

    // Define neural network hyperparameters
    int input_size = 768; // Input size (number of features)
    int hidden_size = 128; // Number of hidden units
    int output_size = 10; // Number of output classes
    int epochs = 30; // Number of training epochs
    float learning_rate = 0.01;

    // Create and train the neural network
    NeuralNetwork* model = createNeuralNetwork(input_size, hidden_size, output_size);
    trainNeuralNetwork(model, &data, epochs, learning_rate);

    InputAndTargets test_data = loadInputAndTargets(test_image_filename, test_label_filename);

    // Test the trained model
    testModel(model, &test_data);

    // Clean up memory
    freeNeuralNetwork(model);
    freeInputAndTargets(&data);
    freeInputAndTargets(&test_data);

    return 0;
}
