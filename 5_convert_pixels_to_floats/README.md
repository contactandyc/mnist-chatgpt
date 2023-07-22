# ChatGPT Prompt

```
write a function which converts an array of uint8_t pixels which range 
from 0 to 255 to floating points that range between 0 and 1 and return the array
```

# ChatGPT Output

Certainly! Here's a function in C that converts an array of `uint8_t` pixels, which range from 0 to 255, to an array of floating points that range between 0 and 1:

```c
#include <stdio.h>
#include <stdlib.h>

float* convertToFloat(uint8_t* pixels, size_t num_pixels) {
    // Allocate memory for the output array
    float* floatPixels = (float*)malloc(num_pixels * sizeof(float));
    if (floatPixels == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        exit(1);
    }

    // Convert each pixel value to a floating point value in the range [0, 1]
    for (size_t i = 0; i < num_pixels; i++) {
        floatPixels[i] = (float)pixels[i] / 255.0;
    }

    return floatPixels;
}

void printFloatPixels(float* floatPixels, size_t num_pixels) {
    for (size_t i = 0; i < num_pixels; i++) {
        printf("%.2f ", floatPixels[i]);
    }
    printf("\n");
}

int main() {
    uint8_t pixels[] = {0, 127, 255};
    size_t num_pixels = sizeof(pixels) / sizeof(pixels[0]);

    float* floatPixels = convertToFloat(pixels, num_pixels);

    // Print the converted floating point pixel values
    printFloatPixels(floatPixels, num_pixels);

    // Free the memory allocated for the floatPixels array
    free(floatPixels);

    return 0;
}
```

The `convertToFloat` function takes an array of `uint8_t` pixels and its size `num_pixels`. It dynamically allocates memory for the output array `floatPixels` and then iterates through each pixel value, converting it to a floating-point value in the range [0, 1]. The converted floating-point pixel values are stored in the `floatPixels` array, which is then returned.

The `printFloatPixels` function is provided for demonstration purposes to print the converted floating-point pixel values.

In the `main` function, we create an example array of `uint8_t` pixels, convert them to floating points, and then print the converted floating-point pixel values. Finally, we free the memory allocated for the `floatPixels` array.

# Summary

The program is converting these pixels

```c
uint8_t pixels[] = {0, 127, 255};
```

Running

```
% ./convert_pixels_to_floats
0.00 0.50 1.00 
```

validates success
