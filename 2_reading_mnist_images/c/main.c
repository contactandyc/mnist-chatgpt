#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <arpa/inet.h> // For big-endian byte order conversion

#define IMAGE_MAGIC_NUMBER 0x00000803

// Structure to store the image data
typedef struct {
    uint32_t magic_number;
    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_columns;
    uint8_t *pixels;
} ImageData;

// Function to read the file and extract the image data
ImageData readImageFile(const char *filename) {
    ImageData data;
    FILE *file = fopen(filename, "rb");

    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }

    // Read the header and convert from big-endian to host byte order
    fread(&data.magic_number, sizeof(uint32_t), 1, file);
    data.magic_number = ntohl(data.magic_number);

    fread(&data.num_images, sizeof(uint32_t), 1, file);
    data.num_images = ntohl(data.num_images);

    fread(&data.num_rows, sizeof(uint32_t), 1, file);
    data.num_rows = ntohl(data.num_rows);

    fread(&data.num_columns, sizeof(uint32_t), 1, file);
    data.num_columns = ntohl(data.num_columns);

    // Check the magic number
    if (data.magic_number != IMAGE_MAGIC_NUMBER) {
        fprintf(stderr, "Invalid magic number in the file: %s\n", filename);
        fclose(file);
        exit(1);
    }

    // Calculate the total number of pixels in the image
    size_t num_pixels = data.num_images * data.num_rows * data.num_columns;
    data.pixels = (uint8_t *)malloc(num_pixels);

    if (data.pixels == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(file);
        exit(1);
    }

    // Read the pixel data
    fread(data.pixels, sizeof(uint8_t), num_pixels, file);

    fclose(file);

    return data;
}

// Function to free the memory allocated for image data
void freeImageData(ImageData data) {
    free(data.pixels);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];
    ImageData imageData = readImageFile(filename);

    // Print the image data or perform other operations with it
    printf("Magic Number: 0x%08x\n", imageData.magic_number);
    printf("Number of Images: %u\n", imageData.num_images);
    printf("Number of Rows: %u\n", imageData.num_rows);
    printf("Number of Columns: %u\n", imageData.num_columns);

    // Example: Print the first 10 pixel values of the first image
    printf("Pixel Values (First 10): ");
    for (size_t i = 0; i < 10; i++) {
        printf("%u ", imageData.pixels[i]);
    }
    printf("\n");

    // Free the memory allocated for image data
    freeImageData(imageData);

    return 0;
}