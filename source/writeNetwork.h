#pragma once

void writeNetwork(char* filename, float* firstLayerWeights, float* firstLayerBiases, int firstLayerCols, int firstLayerRows, 
float* secondLayerWeights, float* secondLayerBiases, int secondLayerCols, int secondLayerRows);
void readNetwork(char* filename, float* firstLayerWeights, float* firstLayerBiases, int* firstLayerCols, int* firstLayerRows, 
float* secondLayerWeights, float* secondLayerBiases, int* secondLayerCols, int* secondLayerRows);