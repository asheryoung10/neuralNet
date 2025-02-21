#include "writeNetwork.h"

#include <stdio.h>
#include <stdlib.h>

void writeNetwork(char* filename,
                  float* firstLayerWeights, float* firstLayerBiases, int firstLayerCols, int firstLayerRows, 
                  float* secondLayerWeights, float* secondLayerBiases, int secondLayerCols, int secondLayerRows) {
    fprintf(stderr, "Opening file.");
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        perror("Error opening file for writing");
        return;
    }
    fprintf(stderr, "File opened.");

    fprintf(stderr, "Writing Header");
    // Write dimensions for first layer: cols and rows.
    fwrite(&firstLayerCols, sizeof(int), 1, fp);
    fwrite(&firstLayerRows, sizeof(int), 1, fp);

    fprintf(stderr, "Writing First Layer");
    // Write the first layer weights and biases.
    size_t count = (size_t)firstLayerCols * firstLayerRows;
    fwrite(firstLayerWeights, sizeof(float), count, fp);
    fwrite(firstLayerBiases, sizeof(float), firstLayerRows, fp);

    fprintf(stderr, "Writing Second Layer");
    // Write dimensions for second layer: cols and rows.
    fwrite(&secondLayerCols, sizeof(int), 1, fp);
    fwrite(&secondLayerRows, sizeof(int), 1, fp);

    // Write the second layer weights and biases.
    count = (size_t)secondLayerCols * secondLayerRows;
    fwrite(secondLayerWeights, sizeof(float), count, fp);
    fwrite(secondLayerBiases, sizeof(float), secondLayerRows, fp);

    fclose(fp);
}

void readNetwork(char* filename,
                 float* firstLayerWeights, float* firstLayerBiases, int* firstLayerCols, int* firstLayerRows, 
                 float* secondLayerWeights, float* secondLayerBiases, int* secondLayerCols, int* secondLayerRows) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Error opening file for reading");
        return;
    }

    // Read dimensions for first layer.
    fread(firstLayerCols, sizeof(int), 1, fp);
    fread(firstLayerRows, sizeof(int), 1, fp);

    // Read first layer weights and biases.
    size_t count = (size_t)(*firstLayerCols) * (*firstLayerRows);
    fread(firstLayerWeights, sizeof(float), count, fp);
    fread(firstLayerBiases, sizeof(float), *firstLayerRows, fp);

    // Read dimensions for second layer.
    fread(secondLayerCols, sizeof(int), 1, fp);
    fread(secondLayerRows, sizeof(int), 1, fp);

    // Read second layer weights and biases.
    count = (size_t)(*secondLayerCols) * (*secondLayerRows);
    fread(secondLayerWeights, sizeof(float), count, fp);
    fread(secondLayerBiases, sizeof(float), *secondLayerRows, fp);

    fclose(fp);
}
