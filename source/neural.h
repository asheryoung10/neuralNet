#pragma once

float* neuralNormalizeImageData(const unsigned char* imageData, int imageDataSize);
void neuralPrintNomalizedImageAndLabel(const float* imageDataNormalized, const unsigned char* labelData, int imagesWidth, int imagesHeight, int imageIndex);

void neuralMatrixMatrixMultiply(const float* matrixA, const float* matrixB, float* matrixResult, int matrixARowCount, int matrixAColumnCount, int matrixBColumnCount);
void neuralMatrixMatrixAdd(const float* matrixA, const float* matrixB, float* matrixResult, int matrixRowCount, int matrixColumnCount);
void neuralVectorVectorAdd(const float* vectorA, const float* vectorB, float* vectorResult, int height);
void neuralMatrixVectorMultiply(const float* matrix, const float* vector, float* result, int matrixRowCount, int matrixColumnCount);
void neuralMatrixTranspose(const float* matrix, float* transpose, int rowCount, int columnCount);

void neuralMatrixInitilizeWeights(float* matrix, int width, int height);
void neuralMatrixInitializeBias(float* matrix, int height);
void neuralVectorApplyRelu(float* vector, int height);
void neuralVectorApplySoftmax(float* vector, int height);
void neuralVectorSetLabel(float* vector, int height, int label);

void neuralComputeOutputError(float* ouputVector, float* trueVector, float* resultVector, int height);
void neuralComputeSecondLayerGradient(float* outputError, float* secondLayerInputVector, float* gradient, int outputErrorHeight, int secondLayerInputVectorHeight);
void neuralComputeFirstLayerOutputError(float* secondLayerWeights, float* outputError, float* firstLayerOutput, float* firstLayerError, int secondLayerWeightsRowCount, int secondLayerWeightsColumnCount);
void neuralComputeFirstLayerGradient(float* firstLayerError, float* inputVector, float* gradient, int firstLayerCols, int firstLayerRows);
void neuralUpdateWeights(float* weightMatrix, float* gradientMatrix, float learningRate, int matrixRowCount, int matrixColumnCount);
void neuralUpdateBiases(float* biasVector, float* gradientVector, float learningRate, int height);

int getPrediction(float* predictionVector, int height);