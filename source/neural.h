#pragma once

float* neuralNormalizeImageData(const unsigned char* imageData, int imageDataSize);
void neuralPrintNomalizedImageAndLabel(const float* imageDataNormalized, const unsigned char* labelData, int imagesWidth, int imagesHeight, int imageIndex);

void neuralMatrixMatrixMultiply(const float* matrixA, const float* matrixB, float* matrixResult, int matrixARowCount, int matrixAColumnCount, int matrixBColumnCount);
void neuralMatrixMatrixAdd(const float* matrixA, const float* matrixB, float* matrixResult, int matrixRowCount, int matrixColumnCount);
void neuralMatrixVectorMultiply(const float* matrix, const float* vector, float* result, int matrixRowCount, int matrixColumnCount);
void neuralMatrixTranspose(const float* matrix, float* transpose, int rowCount, int columnCount);