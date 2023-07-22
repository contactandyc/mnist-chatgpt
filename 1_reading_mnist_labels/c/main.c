#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int magic_number;
    int num_items;
    unsigned char *labels;
} FileResponse;

FileResponse read_file(const char* file_name) {
    FileResponse response;
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

void free_file_response(FileResponse *response) {
    free(response->labels);
}

void process_file(const char* file_name) {
    FileResponse response = read_file(file_name);

    printf("File: %s\n", file_name);
    printf("Magic Number: %d\n", response.magic_number);
    printf("Number of Items: %d\n", response.num_items);

    printf("Labels:\n");
    for (int i = 0; i < response.num_items; i++) {
        printf("[%d] %u\n", i, response.labels[i]);
    }

    printf("\n");

    free_file_response(&response);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s file1 file2 file3 ...\n", argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        process_file(argv[i]);
    }

    return 0;
}