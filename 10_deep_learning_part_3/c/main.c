#include <stdio.h>
#include <stdlib.h>
#include "data_processing.h"
#include "neural_network.h"
#include "activation.h"
#include "ac_allocator.h"

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
    InputAndTargets training_data = loadInputAndTargets(train_image_filename, train_label_filename);
    InputAndTargets test_data = loadInputAndTargets(test_image_filename, test_label_filename);

    /* ChatGPT - please fill in the code here */
    // Create a new NeuralNetwork
    NeuralNetwork* network = createNeuralNetwork(784);

    // Set learning rate and epochs
    float learning_rate = 0.01; // TODO: Adjust as needed
    int epochs = 5; // TODO: Adjust as needed

    // Set the dataset sizes
    Dataset training_dataset = {
        .num_samples = training_data.num_inputs / 100,
        .input_size = 784,
        .output_size = 10,
        .inputs = training_data.inputs,
        .outputs = training_data.targets
    };
    Dataset test_dataset = {
        .num_samples = test_data.num_inputs / 100,
        .input_size = 784,
        .output_size = 10,
        .inputs = test_data.inputs,
        .outputs = test_data.targets
    };

    // Add Dense Layers to the network
    // You can add as many layers as needed and specify the activation function for each layer.
    // For this example, I'm assuming you're using two layers:
    // the first one with a size of 128 and the second one (output layer) with a size of 10 (if you're classifying digits, for instance).
    // I'm also assuming you're using a ReLU activation for the first layer and a Softmax for the output layer.
    // You'll have to implement the corresponding functions: relu, reluDerivative, softmax, softmaxDerivative.
    addDenseLayer(network, 128, sigmoid, sigmoidDerivative);
    addDenseLayer(network, 10, softmax, softmaxDerivative); // If you're classifying digits, the output layer size should be 10.

    // Train the Neural Network
    train(network, &training_dataset, epochs, learning_rate);

    // Test the Neural Network
    test(network, &test_dataset);

    // Don't forget to free your network when you're done
    freeNeuralNetwork(network);

    freeInputAndTargets(&training_data);
    freeInputAndTargets(&test_data);

    return 0;
}
