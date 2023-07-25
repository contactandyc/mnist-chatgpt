#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <stdint.h>

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
ImageData readImageFile(const char *filename);

// Function to free the memory allocated for image data
void freeImageData(ImageData data);

typedef struct {
    int magic_number;
    int num_items;
    unsigned char *labels;
} LabelResponse;

LabelResponse readLabelResponse(const char* file_name);

void freeLabelResponse(LabelResponse *response);

typedef struct {
    int num_targets;
    unsigned char *targets;
    int num_images;
    uint8_t **images;
    uint32_t num_rows;
    uint32_t num_columns;
} ImageDataWithLabels;

// Function to combine image data with label response
ImageDataWithLabels combineImageDataWithLabels(const char *image_file, const char *label_file);

// Function to free the memory allocated for ImageDataWithLabels structure
void freeImageDataWithLabels(ImageDataWithLabels *dataWithLabels);

#endif /* IMAGE_UTILS_H */
