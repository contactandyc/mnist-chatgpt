# My Commentary

In the last section, relu activation caused the deep neural network to have its weights become infinite and nan.

# ChatGPT Prompt

```
It seems that activation functions require special initialization of weights and biases in some cases.  Is this true and is their other things that I should pay attention to?
```

# ChatGPT Output

Yes, that's true. The initialization of the weights in your network can have a significant effect on the training process, and certain initialization strategies can be more effective with certain activation functions. There are a few common strategies that are often used:

1. **Zero Initialization**: This is the simplest type of initialization, where all weights are set to zero. However, this is generally not a good idea because all the neurons in the network will learn the same features during training.

2. **Random Initialization**: Here, weights are assigned random values. This breaks the symmetry and allows neurons to learn different things. However, very large or very small initial values might slow down learning or lead to vanishing/exploding gradients issues.

3. **Xavier / Glorot Initialization**: This method scales the weights according to the number of input and output neurons, which can result in a faster convergence. It is best suited for activation functions that are symmetric around zero and have a range of [-1, 1] or [0, 1], such as Sigmoid or Hyperbolic tangent (tanh).

4. **He Initialization**: This is a variation of Xavier initialization that's scaled differently and is often better suited for ReLU and variants.

In general, it's a good idea to use an initialization method that matches your activation function (e.g., He initialization for ReLU, Xavier initialization for sigmoid or tanh). It's also typically a good idea to initialize biases to small positive values or zero.

It's also important to consider whether you want to use batch normalization, which can help mitigate issues with poor initialization.

The learning rate is another important parameter to consider. If it's too high, the training might be unstable and never converge. If it's too low, the training might be too slow or get stuck in a suboptimal solution. It's often a good idea to start with a somewhat high learning rate and then reduce it over time (this strategy is called learning rate decay).

Regularization (e.g., dropout, weight decay) is another technique to prevent overfitting and improve the generalization of your model. For very deep networks, techniques like residual connections (as in ResNet) can help to alleviate the vanishing/exploding gradient problem.

Lastly, it's a good practice to monitor the training and validation loss during training to diagnose issues such as overfitting or underfitting. You can use techniques such as early stopping to prevent overfitting. If you find that your model is underfitting, you might want to increase its complexity (e.g., add more layers or neurons).

# ChatGPT Prompt

```
Implement each of the initialization functions in C
```

# ChatGPT Output

Sure, here are implementations of the common initialization methods in C:

1. **Zero Initialization**:

```c
void zeroInitialization(float* array, int size) {
    for (int i = 0; i < size; i++) {
        array[i] = 0.0f;
    }
}
```

2. **Random Initialization**:

```c
void randomInitialization(float* array, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        array[i] = ((float) rand() / (RAND_MAX)) - 0.5f;
    }
}
```

3. **Xavier / Glorot Initialization**:

```c
void xavierInitialization(float* array, int size, int num_inputs, int num_outputs) {
    srand(time(NULL));
    float limit = sqrt(6.0f / (num_inputs + num_outputs));
    for (int i = 0; i < size; i++) {
        array[i] = ((float) rand() / (RAND_MAX)) * 2.0f * limit - limit;
    }
}
```

4. **He Initialization**:

```c
void heInitialization(float* array, int size, int num_inputs) {
    srand(time(NULL));
    float limit = sqrt(2.0f / num_inputs);
    for (int i = 0; i < size; i++) {
        array[i] = ((float) rand() / (RAND_MAX)) * 2.0f * limit - limit;
    }
}
```

Each of these functions initializes a given array of a given size. The Xavier and He initialization methods also take the number of input neurons and output neurons into account, as they are used to calculate the initialization range.

For the Xavier initialization, we are using the common "sqrt(6.0 / (n_inputs + n_outputs))" formula for the limit of the uniform distribution, which gives good results when used with the Sigmoid or Tanh activation functions.

For the He initialization, we are using the "sqrt(2.0 / n_inputs)" formula, which works well with ReLU activation functions.

Please note that the C standard library's `rand()` function is a simple pseudo-random number generator and may not be suitable for more complex or secure applications. For such applications, you might want to use a better random number generator.

# ChatGPT Prompt

```
Given a structure

typedef struct {
    int size;
    int inputs_size;
    float* biases;
    float** weights;
    void (*activation_function)(float*, int);
    void (*activation_derivative)(float*, int);
    float* outputs;
} Layer;

How would I use the randomInitialization function above?
```

# ChatGPT Output

The `randomInitialization` function can be used similarly to the previous initialization methods. However, with this initialization method, there's no need for the `inputs_size` field because it doesn't use the layer sizes to scale the random values. Here's how you would use it:

```c
void initializeLayer(Layer* layer) {
    // Initialize biases
    layer->biases = (float*)calloc(layer->size, sizeof(float));

    // Initialize weights
    layer->weights = (float**)malloc(sizeof(float*) * layer->size);
    for (int i = 0; i < layer->size; i++) {
        layer->weights[i] = randomInitialization(layer->inputs_size);
    }
}
```

In this function, the biases are initialized to zeros using the `calloc` function. The weights are initialized by calling `randomInitialization` for each neuron in the layer.

Please note that the randomInitialization function generates random numbers between -1 and 1. You may want to adjust the range of the random values depending on your specific needs.

# ChatGPT Prompt

Given a structure

typedef struct {
int size;
int inputs_size;
float* biases;
float** weights;
void (*activation_function)(float*, int);
void (*activation_derivative)(float*, int);
float* outputs;
} Layer;

How would I use the xavierInitialization function above?

# ChatGPT Output

The `xavierInitialization` function would be used to initialize the weights in each layer. Here is a sample of how it might be used:

```c
void initializeLayer(Layer* layer) {
    // Initialize biases with zeros
    layer->biases = (float*)malloc(layer->size * sizeof(float));
    zeroInitialization(layer->biases, layer->size);

    // Initialize weights with Xavier initialization
    layer->weights = (float**)malloc(layer->size * sizeof(float*));
    for (int i = 0; i < layer->size; i++) {
        layer->weights[i] = (float*)malloc(layer->inputs_size * sizeof(float));
        xavierInitialization(layer->weights[i], layer->inputs_size, layer->inputs_size);
    }
}
```

This `initializeLayer` function accepts a pointer to a `Layer` struct. It then uses `malloc` to allocate memory for the biases and weights in the layer. The biases are initialized to zeros using the `zeroInitialization` function.

For the weights, a loop is used to initialize each row of the weight matrix (which corresponds to the weights for one neuron). Each row is allocated with `malloc` and then initialized with the `xavierInitialization` function. The `xavierInitialization` function is called with the number of inputs to the layer as the `num_inputs` parameter, since the Xavier initialization method is based on the number of inputs to the neuron.

This is a simplified example and may need to be modified to suit your specific needs. Also, please remember to free the memory allocated with `malloc` when it is no longer needed to prevent memory leaks.

# ChatGPT Prompt

```
why is inputs_size used for both the num_inputs and the num_outputs parameters in the xavierInitialization call?
```

# ChatGPT Output

Apologies for the confusion, that was a mistake in my previous explanation.

In the context of a fully connected layer in a neural network, the `num_inputs` would be the size of the previous layer (or the input size for the first layer), and the `num_outputs` would be the size of the current layer.

Here is the corrected code:

```c
void initializeLayer(Layer* layer, int previous_layer_size) {
    // Initialize biases with zeros
    layer->biases = (float*)malloc(layer->size * sizeof(float));
    zeroInitialization(layer->biases, layer->size);

    // Initialize weights with Xavier initialization
    layer->weights = (float**)malloc(layer->size * sizeof(float*));
    for (int i = 0; i < layer->size; i++) {
        layer->weights[i] = (float*)malloc(layer->inputs_size * sizeof(float));
        xavierInitialization(layer->weights[i], previous_layer_size, layer->size);
    }
}
```

In the `xavierInitialization` call, `previous_layer_size` is used for the number of inputs and `layer->size` is used for the number of outputs. The variable `previous_layer_size` represents the size of the layer before the current one (i.e., the number of output neurons from the previous layer which becomes the input for the current layer). If you are initializing the first layer, `previous_layer_size` should be the size of the input to the network.

This adjustment ensures that the weights are initialized appropriately considering both the number of inputs and outputs to the layer, as intended by Xavier initialization.

# ChatGPT Prompt

```
Isn't previous_layer_size layer->inputs_size?
```

# ChatGPT Output

Yes, you're right. In the context of a `Layer` structure where `inputs_size` represents the number of inputs to the layer (which would be the size of the previous layer in a fully connected network), `previous_layer_size` would indeed be the same as `layer->inputs_size`. My apologies for the confusion.

The `initializeLayer` function would then look like this:

```c
void initializeLayer(Layer* layer) {
    // Initialize biases with zeros
    layer->biases = (float*)malloc(layer->size * sizeof(float));
    zeroInitialization(layer->biases, layer->size);

    // Initialize weights with Xavier initialization
    layer->weights = (float**)malloc(layer->size * sizeof(float*));
    for (int i = 0; i < layer->size; i++) {
        layer->weights[i] = (float*)malloc(layer->inputs_size * sizeof(float));
        xavierInitialization(layer->weights[i], layer->inputs_size, layer->size);
    }
}
```

In the `xavierInitialization` call, `layer->inputs_size` is used for the number of inputs and `layer->size` is used for the number of outputs. This reflects the intended use of Xavier initialization which takes into account the number of inputs and outputs to the layer.

# ChatGPT Prompt

```
For Xavier initialization is it typical to have the biases use zero initialization?
```

# ChatGPT Output

Yes, it is typical to initialize the biases as zeros in neural networks, including when using Xavier initialization for the weights.

The main reason for this is that bias terms are responsible for shifting the activation function to the left or right, which can be beneficial when learning complex patterns from the data. If you initialize biases with non-zero values, it might cause the activation function to be shifted too far in one direction from the start, making it harder for the network to learn.

Starting with zero biases ensures that the initial shift is neutral, and the bias terms can adjust appropriately during the learning process. This is not as crucial as the weight initialization, but it is a common practice.

# ChatGPT Prompt

```
Given a structure

typedef struct {
    int size;
    int inputs_size;
    float* biases;
    float** weights;
    void (*activation_function)(float*, int);
    void (*activation_derivative)(float*, int);
    float* outputs;
} Layer;

How would I use the heInitialization function above?
```

# ChatGPT Output

The `heInitialization` function would be used to initialize the weights in each layer. Here is a sample of how it might be used:

```c
void initializeLayer(Layer* layer) {
    // Initialize biases with zeros
    layer->biases = (float*)malloc(layer->size * sizeof(float));
    zeroInitialization(layer->biases, layer->size);

    // Initialize weights with He initialization
    layer->weights = (float**)malloc(layer->size * sizeof(float*));
    for (int i = 0; i < layer->size; i++) {
        layer->weights[i] = (float*)malloc(layer->inputs_size * sizeof(float));
        heInitialization(layer->weights[i], layer->inputs_size, layer->inputs_size);
    }
}
```

This `initializeLayer` function accepts a pointer to a `Layer` struct. It then uses `malloc` to allocate memory for the biases and weights in the layer. The biases are initialized to zeros using the `zeroInitialization` function.

For the weights, a loop is used to initialize each row of the weight matrix (which corresponds to the weights for one neuron). Each row is allocated with `malloc` and then initialized with the `heInitialization` function. The `heInitialization` function is called with the number of inputs to the layer as the `num_inputs` parameter, since the He initialization method is based on the number of inputs to the neuron.

This is a simplified example and may need to be modified to suit your specific needs. Also, please remember to free the memory allocated with `malloc` when it is no longer needed to prevent memory leaks.

# My Commentary

Based upon the above information, all of the biases are zero initialized.  The functions could take the form of the following zeroInitialization function.

```c
float** zeroInitialization(int num_inputs, int num_outputs) {
    float** weights = (float**)malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++)
        weights[i] = (float*)calloc(num_inputs, sizeof(float));
    return weights;
}
```

I would like to put all of the initialization functions in initialization.c/h with the signature above.

# ChatGPT Prompt

```
Can you write a randomInitialization function that looks like

float** zeroInitialization(int num_inputs, int num_outputs) {
    float** weights = (float**)malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++)
        weights[i] = (float*)calloc(num_inputs, sizeof(float));
    return weights;
}
```

# ChatGPT Output

Absolutely! Here's a version of the `randomInitialization` function that initializes a 2D array of weights, similar to the `zeroInitialization` function you provided:

```c
#include <stdlib.h> // for malloc, calloc and rand
#include <time.h>   // for time

float** randomInitialization(int num_inputs, int num_outputs) {
    srand((unsigned int)time(NULL)); // Seed for the random number generator

    float** weights = (float**)malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++) {
        weights[i] = (float*)malloc(num_inputs * sizeof(float));
        for (int j = 0; j < num_inputs; j++) {
            weights[i][j] = (float)rand() / RAND_MAX * 2.0 - 1.0; // Generate random float between -1 and 1
        }
    }
    return weights;
}
```

This function creates a 2D array of weights and fills it with random floating point numbers between -1 and 1. The `rand()` function generates a random integer, which is divided by `RAND_MAX` to create a float between 0 and 1. The resulting value is then scaled and shifted to the range [-1, 1].

Note: It's important to seed the random number generator using `srand()`. If you don't seed the generator, you'll get the same sequence of random numbers every time you run your program. Here, we're using the current time as the seed. This is a common choice, because the current time is always changing. However, the seeding is done every time the function is called which is not a good practice. Ideally, you should seed the random number generator once somewhere at the start of your program.

# ChatGPT Prompt

```
Can you write a xavierInitialization function that looks like

float** zeroInitialization(int num_inputs, int num_outputs) {
    float** weights = (float**)malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++)
        weights[i] = (float*)calloc(num_inputs, sizeof(float));
    return weights;
}
```

# ChatGPT Output

Sure, here is a version of the `xavierInitialization` function that aligns with the style of your `zeroInitialization` function:

```c
#include <stdlib.h> // for malloc, rand
#include <math.h>   // for sqrt
#include <time.h>   // for time

float** xavierInitialization(int num_inputs, int num_outputs) {
    srand((unsigned int)time(NULL)); // Seed for the random number generator

    float** weights = (float**)malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++) {
        weights[i] = (float*)malloc(num_inputs * sizeof(float));
        float std_dev = sqrt(2.0 / (num_inputs + num_outputs));
        for (int j = 0; j < num_inputs; j++) {
            float random_value = (float)rand() / RAND_MAX; // Generate random float between 0 and 1
            weights[i][j] = std_dev * (2.0 * random_value - 1.0); // Generate random float between -std_dev and std_dev
        }
    }
    return weights;
}
```

This version of `xavierInitialization` first calculates the standard deviation `std_dev` using the Xavier initialization formula. It then creates a 2D array of weights and fills it with random floating point numbers between `-std_dev` and `std_dev`.

Note: As in the previous function, it's important to seed the random number generator using `srand()`. This should ideally be done once somewhere at the start of your program.

# ChatGPT Prompt

```
Can you write a heInitialization function that looks like

float** zeroInitialization(int num_inputs, int num_outputs) {
    float** weights = (float**)malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++)
        weights[i] = (float*)calloc(num_inputs, sizeof(float));
    return weights;
}
```

# ChatGPT Output

Sure, here is a version of the `heInitialization` function that aligns with the style of your `zeroInitialization` function:

```c
#include <stdlib.h> // for malloc, rand
#include <math.h>   // for sqrt
#include <time.h>   // for time

float** heInitialization(int num_inputs, int num_outputs) {
    srand((unsigned int)time(NULL)); // Seed for the random number generator

    float** weights = (float**)malloc(num_outputs * sizeof(float*));
    for (int i = 0; i < num_outputs; i++) {
        weights[i] = (float*)malloc(num_inputs * sizeof(float));
        float std_dev = sqrt(2.0 / num_inputs);
        for (int j = 0; j < num_inputs; j++) {
            float random_value = (float)rand() / RAND_MAX; // Generate random float between 0 and 1
            weights[i][j] = std_dev * (2.0 * random_value - 1.0); // Generate random float between -std_dev and std_dev
        }
    }
    return weights;
}
```

This version of `heInitialization` first calculates the standard deviation `std_dev` using the He initialization formula. It then creates a 2D array of weights and fills it with random floating point numbers between `-std_dev` and `std_dev`.

Note: As in the previous function, it's important to seed the random number generator using `srand()`. This should ideally be done once somewhere at the start of your program.

# ChatGPT Prompt

```
Once I put all of the code together in initialization.c/h I ended up with the following header file.

#ifndef _INITIALIZATION_H
#define _INITIALIZATION_H

float** zeroInitialization(int num_inputs, int num_outputs);
float** randomInitialization(int num_inputs, int num_outputs);
float** xavierInitialization(int num_inputs, int num_outputs);
float** heInitialization(int num_inputs, int num_outputs);

#endif /* _INITIALIZATION_H */

Please write comments for this code and regenerate it
```

# ChatGPT Output

Absolutely, here's your revised header file with added comments:

```c
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
```

The comments provide a brief description of each function and its parameters, as well as what the function returns.

# ChatGPT Prompt

```
How would I modify the following code to allow for an initialization function to be passed in as defined above?

typedef struct {
    int size;
    int inputs_size;
    float* biases;
    float** weights;
    float** gradients;
    float** transposed;
    float* output_error;
    void (*activation_function)(float*, int);
    void (*activation_derivative)(float*, int);
    float* outputs;
} Layer;


Layer* addDenseLayer(NeuralNetwork* network, int size, void (*activation_function)(float*, int), void (*activation_derivative)(float*, int)) {
    Layer* layer = (Layer*)ac_malloc(sizeof(Layer));
    int inputs_size;
    if (network->num_layers == 0) {
        inputs_size = network->input_size;
    } else {
        inputs_size = network->layers[network->num_layers-1]->size;
    }

    // Initialize the layer
    layer->size = size;
    layer->inputs_size = inputs_size;
    layer->biases = createRandomArray(size);
    layer->weights = createRandomMatrix(size, inputs_size);
    layer->activation_function = activation_function;
    layer->activation_derivative = activation_derivative;
    layer->outputs = (float*)ac_calloc(size * sizeof(float));
    layer->output_error = (float*)ac_calloc(inputs_size * sizeof(float));
    layer->gradients = createZeroMatrix(size, inputs_size);
    layer->transposed = createZeroMatrix(inputs_size, size);

    // Reallocate memory for the layers array
    network->layers = realloc(network->layers, sizeof(Layer*) * (network->num_layers + 1));

    // Add the new layer to the network
    network->layers[network->num_layers] = layer;
    network->num_layers++;

    return layer;
}
```

# ChatGPT Output

You can modify your `addDenseLayer` function to include an additional argument which is a function pointer to one of the initialization functions. Then, replace the `createRandomMatrix` function call with this function pointer.

Here's how your function definition might look:

```c
Layer* addDenseLayer(NeuralNetwork* network, int size, void (*activation_function)(float*, int), void (*activation_derivative)(float*, int), float** (*initialization_function)(int, int)) {
    Layer* layer = (Layer*)ac_malloc(sizeof(Layer));
    int inputs_size;
    if (network->num_layers == 0) {
        inputs_size = network->input_size;
    } else {
        inputs_size = network->layers[network->num_layers-1]->size;
    }

    // Initialize the layer
    layer->size = size;
    layer->inputs_size = inputs_size;
    layer->biases = createRandomArray(size);
    layer->weights = initialization_function(size, inputs_size);  // Use the passed initialization function
    layer->activation_function = activation_function;
    layer->activation_derivative = activation_derivative;
    layer->outputs = (float*)ac_calloc(size * sizeof(float));
    layer->output_error = (float*)ac_calloc(inputs_size * sizeof(float));
    layer->gradients = createZeroMatrix(size, inputs_size);
    layer->transposed = createZeroMatrix(inputs_size, size);

    // Reallocate memory for the layers array
    network->layers = realloc(network->layers, sizeof(Layer*) * (network->num_layers + 1));

    // Add the new layer to the network
    network->layers[network->num_layers] = layer;
    network->num_layers++;

    return layer;
}
```

Now you can specify the initialization function to use when you add a new layer, for example:

```c
addDenseLayer(network, 128, relu, reluDerivative, heInitialization);
```

Please note that you should make sure the `createRandomArray` function for bias initialization also aligns with your desired initialization strategy. Depending on the initialization strategy, you might want to initialize biases with zeros or with small random numbers.

# My Commentary

The first question prompted multiple answers.  The first was to consider initialization.

The other ideas suggested were to consider
* batch normalization
* the learning rate
* regularization
* monitoring training and validation, possibly using early stopping

What I found after adding heInitialization to relu was that the learning rate needed to be much lower.  Trying a rate of 0.0001 produced much better results than 0.01.  There was also a point when running many epochs where the total loss began to increase or went to nan.  The last idea of early stopping would have helped this.  To do early stopping, it suggests to use a mix of training and validation.

With a learning_rate = 0.01 (in main.c)

```
% ./run.sh
./train_and_test ../../data/train-images-idx3-ubyte ../../data/train-labels-idx1-ubyte ../../data/t10k-images-idx3-ubyte ../../data/t10k-labels-idx1-ubyte
Epoch 1: loss = 0.85563
Epoch 2: loss = nan
```

Updating to 0.001
```
% ./run.sh
./train_and_test ../../data/train-images-idx3-ubyte ../../data/train-labels-idx1-ubyte ../../data/t10k-images-idx3-ubyte ../../data/t10k-labels-idx1-ubyte
Epoch 1: loss = 0.89753
Epoch 2: loss = 0.88873
Epoch 3: loss = 0.87987
Epoch 4: loss = 0.87032
...
Epoch 35: loss = 0.22865
Epoch 36: loss = 0.25295
Epoch 37: loss = 0.30666
Epoch 38: loss = 0.84511
Epoch 39: loss = 1.11938
Epoch 40: loss = 0.90021
Epoch 41: loss = 0.89994
Epoch 42: loss = 0.89972
```

Notice at Epoch 37, the loss begins to increase

Updating to 0.0001
```
% ./run.sh
./train_and_test ../../data/train-images-idx3-ubyte ../../data/train-labels-idx1-ubyte ../../data/t10k-images-idx3-ubyte ../../data/t10k-labels-idx1-ubyte
Epoch 1: loss = 0.90091
Epoch 2: loss = 0.89997
Epoch 3: loss = 0.89905
Epoch 496: loss = 0.07948
Epoch 497: loss = 0.07918
Epoch 498: loss = 0.07887
Epoch 499: loss = 0.07857
Epoch 500: loss = 0.07827
Test Accuracy: 77.40%
```

This completes, but the Test Accuracy doesn't reflect expectations.  The model has likely overfit to the data.  Validation and using more than 1/100th of the training data would likely help this.

Several articles were very helpful to me in exploring this further.

[Intro to optimization in deep learning: Gradient Descent](https://blog.paperspace.com/intro-to-optimization-in-deep-learning-gradient-descent/)
[Intro to Optimization in Deep Learning: Vanishing Gradients and Choosing the Right Activation Function](https://blog.paperspace.com/vanishing-gradients-activation-function/)
