#include <stdio.h>
#include <stdlib.h>
// Function to read a 4-byte integer in big-endian format
int read_int(FILE *file) {
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, file);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

// Function to load MNIST images
unsigned char* load_mnist_images(const char *filename, int *num_images, int *rows, int *cols) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        return NULL;
    }

    // Read header
    int magic_number = read_int(file);
    if (magic_number != 0x00000803) {
        printf("Invalid MNIST image file!\n");
        fclose(file);
        return NULL;
    }

    *num_images = read_int(file);
    *rows = read_int(file);
    *cols = read_int(file);

    printf("Loading %d images of size %dx%d...\n", *num_images, *rows, *cols);

    // Allocate memory for images
    int img_size = (*rows) * (*cols);
    unsigned char *images = (unsigned char*)malloc((*num_images) * img_size);
    fread(images, sizeof(unsigned char), (*num_images) * img_size, file);

    fclose(file);
    return images;
}

// Function to load MNIST labels
unsigned char* load_mnist_labels(const char *filename, int *num_labels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        return NULL;
    }

    // Read header
    int magic_number = read_int(file);
    if (magic_number != 0x00000801) {
        printf("Invalid MNIST label file!\n");
        fclose(file);
        return NULL;
    }

    *num_labels = read_int(file);
    printf("Loading %d labels...\n", *num_labels);

    // Allocate memory for labels
    unsigned char *labels = (unsigned char*)malloc((*num_labels));
    fread(labels, sizeof(unsigned char), (*num_labels), file);

    fclose(file);
    return labels;
}

// Function to print an image as ASCII
void print_mnist_image(unsigned char *image, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%c", image[i * cols + j] > 128 ? '#' : '.'); // Convert to ASCII
        }
        printf("\n");
    }
}
void print_mnist_image_float(float *image, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%c", image[i * cols + j] > 0.3 ? '#' : '.'); // Convert to ASCII
        }
        printf("\n");
    }
}

void normalize_image(unsigned char *images, float *normalized_images, int num_images, int rows, int cols) {
    int img_size = rows * cols;
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < img_size; j++) {
            normalized_images[i * img_size + j] = images[i * img_size + j] / 255.0f; // Normalize
        }
    }
}