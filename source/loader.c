#include <stdio.h>
#include "writeNetwork.h"
#include "neural.h"
#include "datasetReader.h"
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define EPOCH_COUNT 2
#define LEARNING_RATE 0.001
#define FIRST_HIDDEN_LAYER_NEURON_COUNT 512
#define SECOND_HIDDEN_LAYER_NEURON_COUNT 11


int main(int argumentCount, char* arguments[]) {
    if(argumentCount != 5) {
        printf("Please supply location of training image and label dataset files followed by testing image and label dataset files.");
        return 0;
    }

    // Load training data.
    int trainingImageCount, trainingImagesWidth, trainingImagesHeight;
    unsigned char* trainingImages = datasetReaderLoadImages(arguments[1],
        &trainingImageCount, &trainingImagesWidth, &trainingImagesHeight);
    int trainingLabelCount;
    unsigned char* trainingLabels = datasetReaderLoadLabels(arguments[2], &trainingLabelCount);
        
    if(!trainingImages || !trainingLabels) {
        printf("Failed to load image and/or label dataset file(s).");
        return 0;
    } else {
        printf("Training dataset has %d images and %d labels.\n", trainingImageCount, trainingLabelCount);
    }

    float* trainingImagesNormalized = neuralNormalizeImageData(trainingImages, 
        trainingImageCount * trainingImagesWidth * trainingImagesHeight);
    free(trainingImages);
    
    // Load testing data.
    int testingImageCount, testingImagesWidth, testingImagesHeight;
    unsigned char* testingImages = datasetReaderLoadImages(arguments[3],
        &testingImageCount, &testingImagesWidth, &testingImagesHeight);
    int testingLabelCount;
    unsigned char* testingLabels = datasetReaderLoadLabels(arguments[4], &testingLabelCount);
        
    if(!testingImages || !testingLabels) {
        printf("Failed to load image and/or label dataset file(s).");
        return 0;
    } else {
        printf("Testing dataset has %d images and %d labels.\n", testingImageCount, testingLabelCount);
    }

    float* testingImagesNormalized = neuralNormalizeImageData(testingImages, 
        testingImageCount * testingImagesWidth * testingImagesHeight);
    free(testingImages);
    
    // Training
    int firstLayerRowCount = FIRST_HIDDEN_LAYER_NEURON_COUNT;
    int firstLayerColumnCount = trainingImagesWidth * trainingImagesHeight;
    int secondLayerRowCount = SECOND_HIDDEN_LAYER_NEURON_COUNT;
    int secondLayerColumnCount = firstLayerRowCount;
    float* inputVector;
    float* firstIntermediateVector = (float*) malloc(sizeof(float) * firstLayerRowCount);
    float* secondIntermediateVector = (float*) malloc(sizeof(float) * secondLayerRowCount);
    float* firstHiddenLayerWeights = (float*) malloc(sizeof(float) * firstLayerRowCount * firstLayerColumnCount);
    float* firstHiddenLayerBiases = (float*) malloc(sizeof(float) * firstLayerRowCount);
    float* secondHiddenLayerWeights = (float*) malloc(sizeof(float) * secondLayerRowCount * secondLayerColumnCount);
    float* secondHiddenLayerBiases = (float*) malloc(sizeof(float) * secondLayerRowCount);
    float* outputVector = (float*) malloc(sizeof(float) * secondLayerRowCount);
    float* labelVector = (float*) malloc(sizeof(float)  *secondLayerRowCount);
    neuralMatrixInitilizeWeights(firstHiddenLayerWeights, firstLayerColumnCount, firstLayerRowCount);
    neuralMatrixInitilizeWeights(secondHiddenLayerWeights, secondLayerColumnCount, secondLayerRowCount);
    neuralMatrixInitializeBias(firstHiddenLayerBiases, firstLayerRowCount);
    neuralMatrixInitializeBias(secondHiddenLayerBiases, secondLayerRowCount);
    float* outputError = (float*) malloc(sizeof(float) * secondLayerRowCount);
    float* secondLayerWeightGradient = (float*) malloc(sizeof(float) * secondLayerRowCount * secondLayerColumnCount);
    float* secondLayerBiasGradient; // This will just be output error.
    float* firstLayerError = (float*) malloc(sizeof(float) * firstLayerRowCount);
    float* firstLayerWeightGradient = (float*) malloc(sizeof(float) * firstLayerRowCount * firstLayerColumnCount);
    float* firstLayerBiasGradient; // This will just be first layer error.
    readNetwork("networkFile.data", firstHiddenLayerWeights, firstHiddenLayerBiases, &firstLayerColumnCount,
    &firstLayerRowCount, secondHiddenLayerWeights, secondHiddenLayerBiases, &secondLayerColumnCount, &secondLayerRowCount);
    int correctCount = 0;
    for(int image = 0; image < testingImageCount*0; image++) {
          // Forward Propogation
          inputVector = &testingImagesNormalized[image * trainingImagesWidth * trainingImagesHeight];
          neuralMatrixVectorMultiply(firstHiddenLayerWeights, inputVector, firstIntermediateVector, firstLayerColumnCount, firstLayerRowCount);
          neuralVectorVectorAdd(firstIntermediateVector, firstHiddenLayerBiases, firstIntermediateVector, firstLayerRowCount);
          neuralVectorApplyRelu(firstIntermediateVector, firstLayerRowCount);

          neuralMatrixVectorMultiply(secondHiddenLayerWeights, firstIntermediateVector, secondIntermediateVector, secondLayerRowCount, secondLayerColumnCount);
          neuralVectorVectorAdd(secondIntermediateVector, secondHiddenLayerBiases, secondIntermediateVector, secondLayerRowCount);
          neuralVectorApplySoftmax(secondIntermediateVector, secondLayerRowCount);
          
              
          if(getPrediction(secondIntermediateVector, secondLayerRowCount) == (int) testingLabels[image]) {
             correctCount++; 
          }
    }
    printf("Accuracy is %f ", correctCount / (float) testingImageCount);
    int x,y,n;
    unsigned char *data = stbi_load("seven.png", &x, &y, &n, 0);
     inputVector = malloc(sizeof(float) * 28 * 28);
    for(int i = 0; i < x; i++) {
        for(int j = 0; j < y; j++) {
            inputVector[i * y + j] = data[(i * y  + j) * 3] / (float)255.0;
            if(inputVector[i * y + j] > 0.2) {
                printf("#");
            }else {
                printf("o");
            }
        }
        printf("\n");
    }
    neuralMatrixVectorMultiply(firstHiddenLayerWeights, inputVector, firstIntermediateVector, firstLayerColumnCount, firstLayerRowCount);
    neuralVectorVectorAdd(firstIntermediateVector, firstHiddenLayerBiases, firstIntermediateVector, firstLayerRowCount);
    neuralVectorApplyRelu(firstIntermediateVector, firstLayerRowCount);

    neuralMatrixVectorMultiply(secondHiddenLayerWeights, firstIntermediateVector, secondIntermediateVector, secondLayerRowCount, secondLayerColumnCount);
    neuralVectorVectorAdd(secondIntermediateVector, secondHiddenLayerBiases, secondIntermediateVector, secondLayerRowCount);
    neuralVectorApplySoftmax(secondIntermediateVector, secondLayerRowCount);
    printf("My prediction is %d.\n", getPrediction(secondIntermediateVector, secondLayerRowCount));      
              
    
    
}