#pragma once

unsigned char* load_mnist_images(const char *filename, int *num_images, int *rows, int *cols);

unsigned char* load_mnist_labels(const char *filename, int *num_labels);

void print_mnist_image(unsigned char *image, int rows, int cols);

void print_mnist_image_float(float *image, int rows, int cols);

void normalize_image(unsigned char *images, float *normalized_images, int num_images, int rows, int cols);
