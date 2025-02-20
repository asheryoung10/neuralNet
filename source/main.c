#include <stdio.h>
#include "datasetReader.h"
#include "neural.h"
#include <stdlib.h>


int main(int argumentCount, char* arguments[]) {

    if(argumentCount != 3) {
        printf("Please supply location of image and label dataset files.");
        return 0;
    }

    int imageCount, imagesWidth, imagesHeight;
    unsigned char* trainingImages = datasetReaderLoadImages(arguments[1],
        &imageCount, &imagesWidth, &imagesHeight);
    int labelCount;
    unsigned char* trainingLabels = datasetReaderLoadLabels(arguments[2], &labelCount);
        
    if(!trainingImages | !trainingLabels) {
        printf("Failed to load image and/or label dataset file(s).");
        return 0;
    }

    float* trainingImagesNormalized = neuralNormalizeImageData(trainingImages, 
        imageCount * imagesWidth * imagesHeight);
    free(trainingImages);
    neuralPrintNomalizedImageAndLabel(trainingImagesNormalized, 
        trainingLabels, imagesWidth, imagesHeight, 0);
    

    free(trainingImagesNormalized);
    free(trainingLabels);
    return 0;
}