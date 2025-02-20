#include "neural.h"
#include <stdlib.h>
#include <stdio.h>


float* neuralNormalizeImageData(const unsigned char* imageData, int imageDataSize) {
    float* imageDataNormalized = (float*) malloc(sizeof(float) * imageDataSize);
    for(int i = 0; i < imageDataSize; i++) {
        imageDataNormalized[i] = imageData[i] / 255.0f;
    }
    return imageDataNormalized;
}


void neuralPrintNomalizedImageAndLabel(const float* imageDataNormalized, const unsigned char* labelData, int imagesWidth, int imagesHeight, int imageIndex) {
    int imageSize = imagesWidth * imagesHeight;
    int imageOffset = imageSize * imageIndex;
    int label = labelData[imageIndex];
    printf("Printing image #%d. Label is %d.\n", imageIndex + 1, label);
    for(int y = 0; y < imagesWidth; y++) {
        for(int x = 0; x < imagesWidth; x++) {
            if(imageDataNormalized[imageOffset + (y * imagesWidth) + x] > 0.2) {
                printf("#");
            }else {
                printf("O");
            }
        }
        printf("\n");
    }
}


void neuralMatrixMatrixMultiply(const float* matrixA, const float* matrixB, float* matrixResult, int matrixARowCount, int matrixAColumnCount, int matrixBColumnCount) {
    for(int rowA = 0; rowA < matrixARowCount; rowA++) {
        for(int colB = 0; colB < matrixBColumnCount; colB++) {
            float dotProduct = 0;
            for(int colARowB = 0; colARowB < matrixAColumnCount; colARowB++) {
                dotProduct += matrixA[rowA * matrixAColumnCount + colARowB] * matrixB[colARowB * matrixBColumnCount + colB];
            } 
            matrixResult[rowA * matrixBColumnCount + colB] = dotProduct;
        }
    }
}

void neuralMatrixMatrixAdd(const float* matrixA, const float* matrixB, float* matrixResult, int matrixRowCount, int matrixColumnCount) {
    int entryCount = matrixRowCount * matrixColumnCount;
    for(int entry = 0; entry < entryCount; entry++) {
        matrixResult[entry] = matrixA[entry] + matrixB[entry];
    }
}

void neuralMatrixVectorMultiply(const float* matrix, const float* vector, float* result, int matrixRowCount, int matrixColumnCount) {
    for(int row = 0; row < matrixRowCount; row++) {
        float dotProduct = 0;
        for(int column = 0; column < matrixColumnCount; column++) {
            dotProduct += matrix[row * matrixColumnCount + column] * vector[column];
        }
        result[row] = dotProduct;
    }
}


void neuralMatrixTranspose(const float* matrix, float* transpose, int rowCount, int columnCount) {
    for(int row = 0; row < rowCount; row++) {
        for(int column = 0; column < columnCount; column++) {
            transpose[column * rowCount + row] = matrix[row * columnCount + column];
        }
    }
}