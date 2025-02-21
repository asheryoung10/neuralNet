#include "neural.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


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
    for(int y = 0; y < imagesHeight; y++) {
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

void neuralMatrixInitilizeWeights(float* matrix, int width, int height) {
    srand(time(NULL));
    int size = width * height;
    for(int entry = 0; entry < size; entry++) {
        matrix[entry] = ((float) rand() / RAND_MAX - 0.5f) * sqrt(2.0f / width);
    }
}
void neuralMatrixInitializeBias(float* matrix, int height) {
    for(int entry = 0; entry < height; entry++){
        matrix[entry] = 0.0f;
    }
}

void neuralVectorVectorAdd(const float* vectorA, const float* vectorB, float* vectorResult, int height) {
    for(int entry = 0; entry < height; entry++) {
        vectorResult[entry] = vectorA[entry] + vectorB[entry];
    }
}

void neuralVectorApplyRelu(float* vector, int height) {
    for(int  entry = 0; entry < height; entry++) {
        vector[entry] = vector[entry] > 0 ? vector[entry] : 0;
    }
}

void neuralVectorApplySoftmax(float* vector, int height) {
    float max = vector[0];
    for(int entry = 1; entry < height; entry++) {
        if(vector[entry] > max) {
            max = vector[entry];
        }
    }

    float sum = 0;
    for(int entry = 0; entry < height; entry++) {
        vector[entry] = exp(vector[entry] - max);
        sum += vector[entry];
    }

    for(int entry = 0; entry < height; entry++) {
        vector[entry] /= sum;
    }

}


void neuralVectorSetLabel(float* vector, int height, int label) {
    for(int entry = 0; entry < height; entry++) {
        vector[entry] = 0;
    }
    vector[label] = 1;
}


void neuralComputeOutputError(float* ouputVector, float* trueVector, float* resultVector, int height) {
    for(int entry = 0; entry < height; entry++) {
        resultVector[entry] = ouputVector[entry] - trueVector[entry];
    }
}


void neuralComputeSecondLayerGradient(float* outputError, float* secondLayerInputVector, float* gradient, int outputErrorHeight, int secondLayerInputVectorHeight) {
    for(int outputRow = 0; outputRow < outputErrorHeight; outputRow++) {
        for(int secondLayerInputVectorRow = 0; secondLayerInputVectorRow < secondLayerInputVectorHeight; secondLayerInputVectorRow++) {
            gradient[outputRow * secondLayerInputVectorHeight + secondLayerInputVectorRow] = outputError[outputRow] * secondLayerInputVector[secondLayerInputVectorRow];
        }
    }
}


void neuralComputeFirstLayerOutputError(float* secondLayerWeights, float* outputError, float* firstLayerOutput, float* firstLayerError, int secondLayerWeightsRowCount, int secondLayerWeightsColumnCount) {
    for(int columnWeights = 0; columnWeights < secondLayerWeightsColumnCount; columnWeights++) {
        float dotProduct = 0;
        for(int rowWeights = 0; rowWeights < secondLayerWeightsRowCount; rowWeights++) {
            dotProduct += secondLayerWeights[rowWeights * secondLayerWeightsColumnCount + columnWeights] * outputError[rowWeights];
        }
        // Undo Relu.
        firstLayerError[columnWeights] = firstLayerOutput[columnWeights] != 0 ? dotProduct : 0;
    }
}


void neuralComputeFirstLayerGradient(float* firstLayerError, float* inputVector, float* gradient, int firstLayerCols, int firstLayerRows) {
    for(int gradientRow = 0; gradientRow < firstLayerRows; gradientRow++) {
        for(int gradientCol = 0; gradientCol < firstLayerCols; gradientCol++) {
            gradient[gradientRow * firstLayerCols + gradientCol] = firstLayerError[gradientRow] * inputVector[gradientCol];
        }
    }
}

void neuralUpdateWeights(float* weightMatrix, float* gradientMatrix, float learningRate, int matrixRowCount, int matrixColumnCount) {
    int entryCount = matrixColumnCount * matrixRowCount;
    for(int entry = 0; entry < entryCount; entry++) {
        weightMatrix[entry] -= gradientMatrix[entry] * learningRate;
    }
}

void neuralUpdateBiases(float* biasVector, float* gradientVector, float learningRate, int height) {
    for(int entry = 0; entry < height; entry++) {
        biasVector[entry] -= gradientVector[entry] * learningRate;
    }
}

int getPrediction(float* predictionVector, int height) {
    int maxIndex = 0;
    for(int index = 1; index < height; index++){
        if(predictionVector[index] > predictionVector[maxIndex]) {
            maxIndex = index;
        }
    }
    return maxIndex;
}