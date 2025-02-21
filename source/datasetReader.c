#include "datasetReader.h"
#include <stdio.h>
#include <stdlib.h>

int datasetReaderReadInt(FILE* file) {
    unsigned char bytes[4];
    fread(bytes, sizeof(unsigned char), 4, file);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

unsigned char* datasetReaderLoadImages(const char* filename, int* imagesCount, int* imagesWidth, int* imagesHeight) {
    FILE* file = fopen(filename, "rb");
    if(!file) {
        fprintf(stderr, "Failed to read file \"%s\".", filename);
        return NULL;
    }
    // Read header
    int magic_number = datasetReaderReadInt(file);
    if (magic_number != 0x00000803) {
        printf("Invalid file format.\n");
        fclose(file);
        return NULL;
    }

    *imagesCount = datasetReaderReadInt(file);
    *imagesHeight = datasetReaderReadInt(file);
    *imagesWidth = datasetReaderReadInt(file);
    int dataSize = (*imagesCount) * (*imagesWidth) * (*imagesHeight);

    unsigned char* imageData = (unsigned char*) malloc(dataSize);
    fread(imageData, sizeof(unsigned char), (dataSize), file);
    fclose(file);

    return imageData;
}

unsigned char* datasetReaderLoadLabels(const char* filename, int* labelCount) {
    FILE* file = fopen(filename, "rb");
    if(!file) {
        fprintf(stderr, "Failed to read file \"%s\".", filename);
        return NULL;
    }
    // Read header
    int magic_number = datasetReaderReadInt(file);
    if (magic_number != 0x00000801) {
        printf("Invalid file format.\n");
        fclose(file);
        return NULL;
    }

    *labelCount = datasetReaderReadInt(file);
    int dataSize = *labelCount;

    unsigned char* labelData = (unsigned char*) malloc(dataSize);
    fread(labelData, sizeof(unsigned char), dataSize, file);
    fclose(file);

    return labelData;
}

void datasetReaderPrintImageAndLabel(const unsigned char* imageData, const unsigned char* labelData, int imagesWidth, int imagesHeight, int imageIndex) {
    int imageSize = imagesWidth * imagesHeight;
    int imageOffset = imageSize * imageIndex;
    int label = labelData[imageIndex];
    printf("Printing image #%d. Label is %d.\n", imageIndex + 1, label);
    for(int y = 0; y < imagesHeight; y++) {
        for(int x = 0; x < imagesWidth; x++) {
            if(imageData[imageOffset + (y * imagesWidth) + x] > 30) {
                printf("#");
            }else {
                printf("O");
            }
        }
        printf("\n");
    }
}