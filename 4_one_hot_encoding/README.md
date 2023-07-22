# ChatGPT Prompt

```
create a function which converts an unsigned char with values ranging from 
binary 0 to 9 using one hot encoding to return an array of floats and create a makefile for it
```

# ChatGPT Output

Sure! Here's the C code for the function and the corresponding Makefile:

**one_hot_encoding.c**:
```c
#include <stdio.h>
#include <stdlib.h>

float* oneHotEncoding(unsigned char input) {
    // Allocate memory for the output array
    float *output = (float*)malloc(10 * sizeof(float));
    if (output == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Initialize the output array to all zeros
    for (int i = 0; i < 10; i++) {
        output[i] = 0.0;
    }

    // Perform one-hot encoding for the input value
    if (input < 10) {
        output[input] = 1.0;
    } else {
        fprintf(stderr, "Invalid input value: %hhu\n", input);
        exit(1);
    }

    return output;
}

void freeOneHotEncoded(float* encoded) {
    free(encoded);
}
```

**Makefile**:
```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c99

all: one_hot_encoding

one_hot_encoding: one_hot_encoding.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f one_hot_encoding
```

The `oneHotEncoding` function takes an unsigned char `input` with values ranging from 0 to 9. It dynamically allocates memory for the output array, initializes the array to all zeros, and then performs one-hot encoding for the input value. The one-hot encoded result is stored in the `output` array, where the value at the index corresponding to the input value is set to 1.0, and all other values are set to 0.0. The function returns a pointer to the one-hot encoded array.

The `freeOneHotEncoded` function is also provided to free the memory allocated for the one-hot encoded array after it is no longer needed.

The Makefile sets the compiler options and creates the executable `one_hot_encoding` from the `one_hot_encoding.c` source file.

To build the program, create a file named `one_hot_encoding.c` with the provided C code, and then create a file named `Makefile` with the provided Makefile content. Place both files in the same directory and run the `make` command in the terminal. The `one_hot_encoding` executable will be created, and you can use it to test the one-hot encoding functionality.

# My Commentary

Running make produces an error, so back to ChatGPT

# ChatGPT Prompt

```
Running make produces

% make                        
gcc -Wall -Wextra -std=c99 -o one_hot_encoding one_hot_encoding.c
Undefined symbols for architecture x86_64:
  "_main", referenced from:
     implicit entry/start for main executable
ld: symbol(s) not found for architecture x86_64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
make: *** [one_hot_encoding] Error 1
```

# ChatGPT Output

Apologies for the oversight. The error occurs because the source file `one_hot_encoding.c` does not have a `main` function, and the linker cannot find it. The reason is that the provided code is meant to be a library with helper functions for one-hot encoding, and it is not intended to be compiled as a standalone program.

To use the one-hot encoding function from another program, you need to create a separate program that includes the `one_hot_encoding.c` file and calls the `oneHotEncoding` function.

Let's create a new program called `main.c` that uses the one-hot encoding function:

**main.c**:
```c
#include <stdio.h>
#include "one_hot_encoding.h"

int main() {
    unsigned char input = 5;
    float *output = oneHotEncoding(input);

    // Print the one-hot encoded array
    printf("Input %hhu: [", input);
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", output[i]);
    }
    printf("]\n");

    // Free the memory allocated for the one-hot encoded array
    freeOneHotEncoded(output);

    return 0;
}
```

**one_hot_encoding.h**:
```c
#ifndef ONE_HOT_ENCODING_H
#define ONE_HOT_ENCODING_H

// Function to convert an unsigned char value to a one-hot encoded array of floats
float* oneHotEncoding(unsigned char input);

// Function to free the memory allocated for the one-hot encoded array
void freeOneHotEncoded(float* encoded);

#endif /* ONE_HOT_ENCODING_H */
```

**Makefile**:
```makefile
CC = gcc
CFLAGS = -Wall -Wextra -std=c99

all: main

main: main.c one_hot_encoding.c
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f main
```

Now, place all three files (`main.c`, `one_hot_encoding.c`, and `one_hot_encoding.h`) in the same directory and run `make` again. This time, it will create an executable called `main`, which will use the one-hot encoding function to print the one-hot encoded array for the input value.

# My Commentary

The above worked, but it named the program `main`.  I placed the code in attempt2 and then modified it to call the program `one_hot_encoding` in the `c` directory.

```bash
c % make
gcc -Wall -Wextra -std=c99 -o one_hot_encoding main.c one_hot_encoding.c
```

```bash
% ./one_hot_encoding 
Input 5: [0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 ]
```
