#pragma once

unsigned char* datasetReaderLoadImages(const char* filename, int* imageCount, int* imagesWidth, int* imagesHeight);
unsigned char* datasetReaderLoadLabels(const char* filename, int* labelCount);
void datasetReaderPrintImageAndLabel(const unsigned char* imageData, const unsigned char* labelData, int imagesWidth, int imagesHeight, int imageIndex);