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
    uint8_t **images; // Array of pointers to image pixel data
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

    // Allocate memory for the array of image pixel data pointers
    data.images = (uint8_t **)malloc(data.num_images * sizeof(uint8_t *));
    if (data.images == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        fclose(file);
        exit(1);
    }

    // Calculate the total number of pixels in each image
    size_t num_pixels = data.num_rows * data.num_columns;

    // Read the pixel data for each image
    for (uint32_t i = 0; i < data.num_images; i++) {
        data.images[i] = (uint8_t *)malloc(num_pixels * sizeof(uint8_t));
        if (data.images[i] == NULL) {
            fprintf(stderr, "Memory allocation error\n");
            fclose(file);
            exit(1);
        }
        fread(data.images[i], sizeof(uint8_t), num_pixels, file);
    }

    fclose(file);

    return data;
}

// Function to free the memory allocated for image data
void freeImageData(ImageData data) {
    for (uint32_t i = 0; i < data.num_images; i++) {
        free(data.images[i]);
    }
    free(data.images);
}

typedef struct {
    int magic_number;
    int num_items;
    unsigned char *labels;
} LabelResponse;

LabelResponse readLabelResponse(const char* file_name) {
    LabelResponse response;
    FILE* file = fopen(file_name, "rb");

    if (file == NULL) {
        printf("Error opening file.\n");
        exit(1);
    }

    // Read magic number
    fread(&response.magic_number, sizeof(int), 1, file);
    response.magic_number = htonl(response.magic_number); // Convert to big-endian if necessary

    // Read number of items
    fread(&response.num_items, sizeof(int), 1, file);
    response.num_items = htonl(response.num_items); // Convert to big-endian if necessary

    // Allocate memory for the labels
    response.labels = (unsigned char*)malloc(response.num_items * sizeof(unsigned char));
    if (response.labels == NULL) {
        printf("Memory allocation error.\n");
        fclose(file);
        exit(1);
    }

    // Read labels
    fread(response.labels, sizeof(unsigned char), response.num_items, file);

    fclose(file);
    return response;
}

void freeLabelResponse(LabelResponse *response) {
    free(response->labels);
}

typedef struct {
    int num_targets;
    unsigned char *targets;
    int num_images;
    uint8_t **images;
} ImageDataWithLabels;

// Function to combine image data with label response
ImageDataWithLabels combineImageDataWithLabels(const char *image_file, const char *label_file) {
    ImageDataWithLabels dataWithLabels;
    LabelResponse labelResponse;

    // Read image data
    ImageData imageData = readImageFile(image_file);

    // Read label response
    labelResponse = readLabelResponse(label_file);

    // Check if the number of images and labels match
    if (imageData.num_images != labelResponse.num_items) {
        printf("Error: The number of images and labels do not match.\n");
        freeImageData(imageData);
        freeLabelResponse(&labelResponse);
        exit(1);
    }

    // Allocate memory for the targets
    dataWithLabels.num_targets = labelResponse.num_items;
    dataWithLabels.targets = (unsigned char *)malloc(dataWithLabels.num_targets * sizeof(unsigned char));
    if (dataWithLabels.targets == NULL) {
        printf("Memory allocation error.\n");
        freeImageData(imageData);
        freeLabelResponse(&labelResponse);
        exit(1);
    }

    // Copy the labels to the targets array
    for (int i = 0; i < dataWithLabels.num_targets; i++) {
        dataWithLabels.targets[i] = labelResponse.labels[i];
    }

    // Copy image data to the new structure
    dataWithLabels.num_images = imageData.num_images;
    dataWithLabels.images = imageData.images;

    // Free the image data structure (without freeing the pixel data pointers, as they are now used by dataWithLabels)
    free(imageData);

    // Free the label response data
    freeLabelResponse(&labelResponse);

    return dataWithLabels;
}

// Function to free the memory allocated for ImageDataWithLabels structure
void freeImageDataWithLabels(ImageDataWithLabels *dataWithLabels) {
    free(dataWithLabels->targets);
    // The pixel data pointers (dataWithLabels->images) are not freed here because they are already freed when calling freeImageData
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <image_filename> <label_filename>\n", argv[0]);
        return 1;
    }

    const char *image_filename = argv[1];
    const char *label_filename = argv[2];

    // Combine image data with label response
    ImageDataWithLabels dataWithLabels = combineImageDataWithLabels(image_filename, label_filename);

    // Print or use the data as needed (e.g., print the first 10 targets and the first 20 images)
    printf("Number of Targets: %d\n", dataWithLabels.num_targets);
    printf("Number of Images: %d\n", dataWithLabels.num_images);

    printf("\nFirst 10 Targets: ");
    for (int i = 0; i < 10 && i < dataWithLabels.num_targets; i++) {
        printf("%d ", dataWithLabels.targets[i]);
    }
    printf("\n");

    printf("\nFirst 20 Images:\n");
    for (int i = 0; i < 20 && i < dataWithLabels.num_images; i++) {
        printf("Image %d:\n", i + 1);
        for (size_t j = 0; j < 20 && j < dataWithLabels.num_rows * dataWithLabels.num_columns; j++) {
            printf("%u ", dataWithLabels.images[i][j]);
        }
        printf("\n");
    }

    // Free the memory allocated for ImageDataWithLabels structure
    freeImageDataWithLabels(&dataWithLabels);

    return 0;
}