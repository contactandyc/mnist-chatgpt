
At this point, I'm going to go for a couple of new tools.  The first is valgrind.

```bash
% brew install valgrind
Running `brew update --auto-update`...
==> Homebrew collects anonymous analytics.
Read the analytics documentation (and how to opt-out) here:
  https://docs.brew.sh/Analytics
No analytics have been recorded yet (nor will be during this `brew` run).

Installing from the API is now the default behaviour!
You can save space and time by running:
  brew untap homebrew/core
==> Auto-updated Homebrew!
Updated 3 taps (hashicorp/tap, homebrew/core and homebrew/cask).
==> New Formulae
bfs llm mvfst plog python-cryptography terragrunt-atlantis-config tzdiff
==> New Casks
herd keyclu lm-studio ripx

You have 12 outdated formulae installed.

valgrind: Linux is required for this software.
Error: valgrind: An unsatisfied requirement failed this build.
```

valgrind is apparently not available for Mac OS X (13.4.1)

# ChatGPT Prompt

```
How can I install valgrind on Mac OS X 13.4.1?
```

As of my last update in September 2021, installing Valgrind on macOS was sometimes a bit tricky because of compatibility issues with certain macOS versions, especially the later ones.

One common method to install Valgrind is via Homebrew. Homebrew is a package manager for macOS that can be used to install a wide variety of software. Here's the general procedure:

1. First, you need to install Homebrew if it's not already installed. You can do this by opening the Terminal application and pasting the following command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Once Homebrew is installed, you can install Valgrind with the following command:

```bash
brew install valgrind
```

However, as mentioned, as of my knowledge cut-off in September 2021, Valgrind had some compatibility issues with macOS, especially versions later than Mojave (10.14). If this is still the case, you might run into problems installing or running Valgrind on macOS 13.4.1.

Alternative tools you might consider include:

- **LLDB** with AddressSanitizer: This combination can often serve a similar purpose to Valgrind.
- **Instruments**: This is a performance, analysis, and testing tool for macOS that is part of Xcode.

Please verify the current compatibility status of Valgrind with your version of macOS from updated sources or consider using the alternatives.

# ChatGPT Prompt

Going with ChatGPT's suggestion for now

```
How do I use LLDB with AddressSanitizer on a Mac?
```

# ChatGPT Output

LLDB is the default debugger in Xcode on macOS and is compatible with AddressSanitizer. AddressSanitizer is a fast memory error detector built into LLVM/Clang, GCC, and other compilers. It can detect several types of memory-related errors, including use-after-free and memory leaks.

Here's how you can use AddressSanitizer with LLDB in Xcode:

1. **Enable AddressSanitizer in Xcode**:

    - Open your project in Xcode.
    - Select the active scheme in the toolbar, then choose Edit Scheme.
    - In the dialog that appears, select the Run phase in the left column.
    - In the Info tab, check the Enable Address Sanitizer checkbox.

2. **Run your program**:

    - Build and run your program as usual. If AddressSanitizer detects a memory error, it will halt the program, print a report to the console, and pause the program in the debugger.

3. **Analyze the error with LLDB**:

    - If a memory error is detected, you can use LLDB as usual to analyze the state of your program. For example, you can use the `bt` command to print a backtrace, or the `frame variable` command to inspect local variables.

4. **Interpreting AddressSanitizer reports**:

    - AddressSanitizer reports contain a lot of useful information, including the type of error (e.g., use-after-free), the location of the error (including a stack trace), and the memory address involved.

5. **Disable AddressSanitizer**:

    - Once you're done debugging, don't forget to disable AddressSanitizer for normal development and performance testing. You can do this by unchecking the Enable Address Sanitizer checkbox in the scheme editor.

Remember, AddressSanitizer only reports the first memory error it encounters. If your program has multiple memory errors, you'll need to fix them one at a time, each time re-running your program with AddressSanitizer enabled.

# ChatGPT Prompt

I'd like to use this on the command line

```
Can AddressSanitizer be enabled in a Makefile for a c project outside of Xcode?
```

# ChatGPT Output

Yes, AddressSanitizer can be used outside of Xcode in a Makefile for a C project. You can add the necessary compiler and linker flags to your `gcc` or `clang` commands in your Makefile.

Here's how you can modify a simple Makefile to use AddressSanitizer:

```Makefile
CC=gcc  # or clang, if you prefer
CFLAGS=-g -fsanitize=address
LDFLAGS=-fsanitize=address

all: your_program

your_program: your_program.o
	$(CC) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -f *.o your_program
```

In this Makefile:

- `CC` is the C compiler command.
- `CFLAGS` are the compiler flags.
- `LDFLAGS` are the linker flags.
- The `-g` option in `CFLAGS` includes debugging information in the compiled program (optional but helpful when debugging).
- The `-fsanitize=address` option enables AddressSanitizer.

This will compile `your_program.c` to an executable named `your_program` with AddressSanitizer enabled. If `your_program` contains a memory error, AddressSanitizer will report it when you run the program.

Please replace `your_program` and `your_program.o` with your actual `.c` and `.o` files. If you have multiple `.c` files, you need to add each corresponding `.o` file to the dependencies of the `your_program` rule.

Remember, AddressSanitizer can cause your program to use more memory and CPU time than usual, so you may want to disable it for normal use or performance testing.

# My Commentary

I changed the main function reducing the number of samples in the train and test by 100.

```c
    // Set the dataset sizes
    Dataset training_dataset = {
        .num_samples = training_data.num_inputs,
   ...

    Dataset test_dataset = {
        .num_samples = test_data.num_inputs,
    ...
```

became

```c
    // Set the dataset sizes
    Dataset training_dataset = {
        .num_samples = training_data.num_inputs / 100,
   ...

    Dataset test_dataset = {
        .num_samples = test_data.num_inputs / 100,
    ...
```

I applied the fsanitize=address option to the Makefile in attempt1, compiled, and ran

```bash
% make clean; make                                                                                                                                                           
rm -f main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o train_and_test
gcc -g -fsanitize=address -I. -c main.c -o main.o
gcc -g -fsanitize=address -I. -c activation.c -o activation.o
gcc -g -fsanitize=address -I. -c matrix.c -o matrix.o
gcc -g -fsanitize=address -I. -c image_utils.c -o image_utils.o
gcc -g -fsanitize=address -I. -c data_processing.c -o data_processing.o
gcc -g -fsanitize=address -I. -c neural_network.c -o neural_network.o
gcc -g -fsanitize=address -I. -o train_and_test main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o
ac@MacBook-Pro attempt1 % lldb ./train_and_test ../../../data/train-images-idx3-ubyte ../../../data/train-labels-idx1-ubyte ../../../data/t10k-images-idx3-ubyte ../../../data/t10k-labels-idx1-ubyte
(lldb) target create "./train_and_test"
Current executable set to '/Users/ac/code/neural/9_deep_learning_part_2/c/attempt1/train_and_test' (x86_64).
(lldb) settings set -- target.run-args  "../../../data/train-images-idx3-ubyte" "../../../data/train-labels-idx1-ubyte" "../../../data/t10k-images-idx3-ubyte" "../../../data/t10k-labels-idx1-ubyte"
(lldb) r
Process 80035 launched: '/Users/ac/code/neural/9_deep_learning_part_2/c/attempt1/train_and_test' (x86_64)
train_and_test(80035,0x7ff848c26640) malloc: nano zone abandoned due to inability to reserve vm space.
=================================================================
==80035==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x604000446278 at pc 0x000100004554 bp 0x7ff7bfefebd0 sp 0x7ff7bfefebc8
READ of size 4 at 0x604000446278 thread T0
    #0 0x100004553 in outerProduct matrix.c:43
    #1 0x10000a6af in backwardPass neural_network.c:93
    #2 0x10000afa0 in train neural_network.c:122
    #3 0x100003634 in main main.c:56
    #4 0x7ff8051f041e in start+0x76e (dyld:x86_64+0xfffffffffff6e41e) (BuildId: 5db85b72c63a318291e55c942ec30e4832000000200000000100000000040d00)

0x604000446278 is located 0 bytes to the right of 40-byte region [0x604000446250,0x604000446278)
allocated by thread T0 here:
    #0 0x1003e8da0 in wrap_malloc+0xa0 (libclang_rt.asan_osx_dynamic.dylib:x86_64h+0x48da0) (BuildId: 756bb7515781379f84412f22c4274ffd2400000010000000000a0a0000030d00)
    #1 0x100003aae in softmax activation.c:26
    #2 0x100009f78 in forwardPass neural_network.c:71
    #3 0x10000ae0c in train neural_network.c:116
    #4 0x100003634 in main main.c:56
    #5 0x7ff8051f041e in start+0x76e (dyld:x86_64+0xfffffffffff6e41e) (BuildId: 5db85b72c63a318291e55c942ec30e4832000000200000000100000000040d00)

SUMMARY: AddressSanitizer: heap-buffer-overflow matrix.c:43 in outerProduct
Shadow bytes around the buggy address:
  0x1c0800088bf0: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
  0x1c0800088c00: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
  0x1c0800088c10: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
  0x1c0800088c20: fa fa 00 00 00 00 00 00 fa fa 00 00 00 00 00 00
  0x1c0800088c30: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
=>0x1c0800088c40: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00[fa]
  0x1c0800088c50: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
  0x1c0800088c60: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
  0x1c0800088c70: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
  0x1c0800088c80: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
  0x1c0800088c90: fa fa 00 00 00 00 00 fa fa fa 00 00 00 00 00 fa
Shadow byte legend (one shadow byte represents 8 application bytes):
  Addressable:           00
  Partially addressable: 01 02 03 04 05 06 07 
  Heap left redzone:       fa
  Freed heap region:       fd
  Stack left redzone:      f1
  Stack mid redzone:       f2
  Stack right redzone:     f3
  Stack after return:      f5
  Stack use after scope:   f8
  Global redzone:          f9
  Global init order:       f6
  Poisoned by user:        f7
  Container overflow:      fc
  Array cookie:            ac
  Intra object redzone:    bb
  ASan internal:           fe
  Left alloca redzone:     ca
  Right alloca redzone:    cb
==80035==ABORTING
(lldb) AddressSanitizer report breakpoint hit. Use 'thread info -s' to get extended information about the report.
Process 80035 stopped
* thread #1, queue = 'com.apple.main-thread', stop reason = Heap buffer overflow
    frame #0: 0x00000001003f3b40 libclang_rt.asan_osx_dynamic.dylib`__asan::AsanDie()
libclang_rt.asan_osx_dynamic.dylib`__asan::AsanDie:
->  0x1003f3b40 <+0>: pushq  %rbp
    0x1003f3b41 <+1>: movq   %rsp, %rbp
    0x1003f3b44 <+4>: pushq  %rbx
    0x1003f3b45 <+5>: pushq  %rax
Target 0: (train_and_test) stopped.
(lldb)  bt
* thread #1, queue = 'com.apple.main-thread', stop reason = Heap buffer overflow
  * frame #0: 0x00000001003f3b40 libclang_rt.asan_osx_dynamic.dylib`__asan::AsanDie()
    frame #1: 0x000000010040e07f libclang_rt.asan_osx_dynamic.dylib`__sanitizer::Die() + 175
    frame #2: 0x00000001003f1da7 libclang_rt.asan_osx_dynamic.dylib`__asan::ScopedInErrorReport::~ScopedInErrorReport() + 1191
    frame #3: 0x00000001003f104c libclang_rt.asan_osx_dynamic.dylib`__asan::ReportGenericError(unsigned long, unsigned long, unsigned long, unsigned long, bool, unsigned long, unsigned int, bool) + 1772
    frame #4: 0x00000001003f22c8 libclang_rt.asan_osx_dynamic.dylib`__asan_report_load4 + 40
    frame #5: 0x0000000100004554 train_and_test`outerProduct(a=0x0000615000004180, b=0x0000604000446250, a_len=128, b_len=784) at matrix.c:43:35
    frame #6: 0x000000010000a6b0 train_and_test`backwardPass(network=0x00006020000000d0, error=0x0000604000446290, learning_rate=0.00999999977) at neural_network.c:93:36
    frame #7: 0x000000010000afa1 train_and_test`train(network=0x00006020000000d0, data=0x00007ff7bfeff080, epochs=5, learning_rate=0.00999999977) at neural_network.c:122:13
    frame #8: 0x0000000100003635 train_and_test`main(argc=5, argv=0x00007ff7bfeff540) at main.c:56:5
    frame #9: 0x00007ff8051f041f dyld`start + 1903
(lldb
```

This gave me the general area where the problem was.  I was able to see fairly quickly that the problems.

```c
float* layer_inputs = i == 0 ? output_layer->outputs : network->layers[i-1]->outputs;
```

is choosing the final output layer instead of the input in backwards propagation.  To fix this, I pass input to backwardPass and am able to remove output_layer variable altogether.

```c
void backwardPass(NeuralNetwork* network, float* error, float learning_rate) {
    Layer* output_layer = network->layers[network->num_layers-1];
    // Compute the error derivative of the output layer
    float* output_error = error;
    for (int i = network->num_layers-1; i >= 0; i--) {
        Layer* layer = network->layers[i];
        float* layer_inputs = i == 0 ? output_layer->outputs : network->layers[i-1]->outputs;  // modified line
```

becomes
```c
void backwardPass(NeuralNetwork* network, float *input, float* error, float learning_rate) {
    // Compute the error derivative of the output layer
    float* output_error = error;
    for (int i = network->num_layers-1; i >= 0; i--) {
        Layer* layer = network->layers[i];
        float* layer_inputs = i == 0 ? input : network->layers[i-1]->outputs;  // modified line
```

I also noticed that the output should not be free in train or test, so I commented those out as well.

```c
free(output);
```

becomes

```c
// free(output);
```

in train and test function.

When I make and run the program in attempt2, the program runs, but the output is unexpected.

```bash
% make clean; make 
rm -f main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o train_and_test
gcc -Wall -Wextra -std=c99 -g -I. -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -g -I. -c activation.c -o activation.o
gcc -Wall -Wextra -std=c99 -g -I. -c matrix.c -o matrix.o
gcc -Wall -Wextra -std=c99 -g -I. -c image_utils.c -o image_utils.o
gcc -Wall -Wextra -std=c99 -g -I. -c data_processing.c -o data_processing.o
gcc -Wall -Wextra -std=c99 -g -I. -c neural_network.c -o neural_network.o
gcc -Wall -Wextra -std=c99 -g -I. -o train_and_test main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o
% lldb ./train_and_test ../../../data/train-images-idx3-ubyte ../../../data/train-labels-idx1-ubyte ../../../data/t10k-images-idx3-ubyte ../../../data/t10k-labels-idx1-ubyte
(lldb) target create "./train_and_test"
Current executable set to '/Users/ac/code/neural/9_deep_learning_part_2/c/attempt2/train_and_test' (x86_64).
(lldb) settings set -- target.run-args  "../../../data/train-images-idx3-ubyte" "../../../data/train-labels-idx1-ubyte" "../../../data/t10k-images-idx3-ubyte" "../../../data/t10k-labels-idx1-ubyte"
(lldb) r
Process 80337 launched: '/Users/ac/code/neural/9_deep_learning_part_2/c/attempt2/train_and_test' (x86_64)
Epoch 1: loss = nan
Epoch 2: loss = nan
Epoch 3: loss = nan
Epoch 4: loss = nan
Epoch 5: loss = nan
Test Accuracy: 8.00%
Process 80337 exited with status = 0 (0x00000000) 
```

# My Commentary

One reason nan can occur is because the activation and the derivative functions require that the weights are initially defined in a certain way so that the weights don't become nan or infinite.  reLu works better a different randomization that what is provided in the code.  To test this theory, I asked chatGPT to produce the sigmoid and derivative functions and try this.

# ChatGPT Prompt
 
```
can you provide a sigmoid and sigmoidDerivative function that is the same style as the reLu function below?

// ReLU activation function
float* relu(float* inputs, int size) {
    float* outputs = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
    }
    return outputs;
}
```

# ChatGPT Output

Sure! Below are the sigmoid function and its derivative:

```c
#include <math.h>

// Sigmoid activation function
float* sigmoid(float* inputs, int size) {
    float* outputs = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        outputs[i] = 1.0 / (1.0 + exp(-inputs[i]));
    }
    return outputs;
}

// Derivative of the sigmoid function
float* sigmoidDerivative(float* inputs, int size) {
    float* outputs = sigmoid(inputs, size); // get sigmoid of inputs
    for (int i = 0; i < size; i++) {
        outputs[i] = outputs[i] * (1 - outputs[i]);
    }
    return outputs;
}
```

In the sigmoid function, we apply the sigmoid operation to each input.

In the sigmoid derivative function, we first apply the sigmoid operation to the inputs. Then, for each element in the output array, we multiply the element by (1 - element) to get the derivative value. The reason for this is that the derivative of the sigmoid function is `sigmoid(x) * (1 - sigmoid(x))`, where `sigmoid(x)` is the sigmoid of the input `x`.

# My Commentary

In attempt3, I add those functions to the activation.c/h and change the reLu to sigmoid in main.c
```
% make clean; make
rm -f main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o train_and_test
gcc -Wall -Wextra -std=c99 -g -I. -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -g -I. -c activation.c -o activation.o
gcc -Wall -Wextra -std=c99 -g -I. -c matrix.c -o matrix.o
gcc -Wall -Wextra -std=c99 -g -I. -c image_utils.c -o image_utils.o
gcc -Wall -Wextra -std=c99 -g -I. -c data_processing.c -o data_processing.o
gcc -Wall -Wextra -std=c99 -g -I. -c neural_network.c -o neural_network.o
gcc -Wall -Wextra -std=c99 -g -I. -o train_and_test main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o
% lldb ./train_and_test ../../../data/train-images-idx3-ubyte ../../../data/train-labels-idx1-ubyte ../../../data/t10k-images-idx3-ubyte ../../../data/t10k-labels-idx1-ubyte
(lldb) target create "./train_and_test"
Current executable set to '/Users/ac/code/neural/9_deep_learning_part_2/c/attempt3/train_and_test' (x86_64).
(lldb) settings set -- target.run-args  "../../../data/train-images-idx3-ubyte" "../../../data/train-labels-idx1-ubyte" "../../../data/t10k-images-idx3-ubyte" "../../../data/t10k-labels-idx1-ubyte"
(lldb) r
Process 80603 launched: '/Users/ac/code/neural/9_deep_learning_part_2/c/attempt3/train_and_test' (x86_64)
Epoch 1: loss = 0.65101
Epoch 2: loss = 0.40358
Epoch 3: loss = 0.36482
Epoch 4: loss = 0.34977
Epoch 5: loss = 0.34126
Test Accuracy: 75.00%
Process 80603 exited with status = 0 (0x00000000)
(lldb) 
```

It appears to be working!  I will add another layer in attempt4

main.c
```c
    addDenseLayer(network, 128, sigmoid, sigmoidDerivative);
    addDenseLayer(network, 10, softmax, softmaxDerivative); // If you're classifying digits, the output layer size should be 10.
```

becomes

```c
    addDenseLayer(network, 128, sigmoid, sigmoidDerivative);
    addDenseLayer(network, 256, sigmoid, sigmoidDerivative);
    addDenseLayer(network, 64, sigmoid, sigmoidDerivative);
    addDenseLayer(network, 10, softmax, softmaxDerivative); // If you're classifying digits, the output layer size should be 10.
```

```bash
% make clean;make                                                                                                                                                            
rm -f main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o train_and_test
gcc -Wall -Wextra -std=c99 -g -I. -c main.c -o main.o
gcc -Wall -Wextra -std=c99 -g -I. -c activation.c -o activation.o
gcc -Wall -Wextra -std=c99 -g -I. -c matrix.c -o matrix.o
gcc -Wall -Wextra -std=c99 -g -I. -c image_utils.c -o image_utils.o
gcc -Wall -Wextra -std=c99 -g -I. -c data_processing.c -o data_processing.o
gcc -Wall -Wextra -std=c99 -g -I. -c neural_network.c -o neural_network.o
gcc -Wall -Wextra -std=c99 -g -I. -o train_and_test main.o activation.o matrix.o image_utils.o data_processing.o neural_network.o
% lldb ./train_and_test ../../../data/train-images-idx3-ubyte ../../../data/train-labels-idx1-ubyte ../../../data/t10k-images-idx3-ubyte ../../../data/t10k-labels-idx1-ubyte
(lldb) target create "./train_and_test"
Current executable set to '/Users/ac/code/neural/9_deep_learning_part_2/c/attempt4/train_and_test' (x86_64).
(lldb) settings set -- target.run-args  "../../../data/train-images-idx3-ubyte" "../../../data/train-labels-idx1-ubyte" "../../../data/t10k-images-idx3-ubyte" "../../../data/t10k-labels-idx1-ubyte"
(lldb) r
Process 80908 launched: '/Users/ac/code/neural/9_deep_learning_part_2/c/attempt4/train_and_test' (x86_64)
Epoch 1: loss = 0.86514
Epoch 2: loss = 0.80352
Epoch 3: loss = 0.73679
Epoch 4: loss = 0.72054
Epoch 5: loss = 0.70861
Test Accuracy: 45.00%
Process 80908 exited with status = 0 (0x00000000) 
(lldb)
```

In this case more layers don't help.  I also noticed that repeated runs would produce different accuracy.  From what I understand, neural networks should be seeded with random values.  However, to make things consistent, I've also commented out the srand calls in neural_network.c

