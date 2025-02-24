#include <stdio.h>
#include "datasetReader.h"
#include "neural.h"
#include <stdlib.h>
#include "writeNetwork.h"

#define EPOCH_COUNT 2
#define LEARNING_RATE 0.002
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
    for(int epoch = 0; epoch < EPOCH_COUNT; epoch++) {
        printf("Training Epoch %d/%d:\n", epoch + 1, EPOCH_COUNT);
        int correctCount = 0;
        for(int image = 0; image < trainingImageCount; image++) {
     
            if( (image+1) % 10000 == 0) {
                printf("\t Image %d/%d\n", image+1, trainingImageCount);
            }
            // Forward Propogation
            inputVector = &trainingImagesNormalized[image * trainingImagesWidth * trainingImagesHeight];
            neuralMatrixVectorMultiply(firstHiddenLayerWeights, inputVector, firstIntermediateVector, firstLayerColumnCount, firstLayerRowCount);
            neuralVectorVectorAdd(firstIntermediateVector, firstHiddenLayerBiases, firstIntermediateVector, firstLayerRowCount);
            neuralVectorApplyRelu(firstIntermediateVector, firstLayerRowCount);

            neuralMatrixVectorMultiply(secondHiddenLayerWeights, firstIntermediateVector, secondIntermediateVector, secondLayerRowCount, secondLayerColumnCount);
            neuralVectorVectorAdd(secondIntermediateVector, secondHiddenLayerBiases, secondIntermediateVector, secondLayerRowCount);
            neuralVectorApplySoftmax(secondIntermediateVector, secondLayerRowCount);
            
                
            if(getPrediction(secondIntermediateVector, secondLayerRowCount) == (int) trainingLabels[image]) {
               correctCount++; 
            }
          
            // Backward Propogation.
            // Find second layer gradients.
            neuralVectorSetLabel(labelVector, secondLayerRowCount, trainingLabels[image]);
            neuralComputeOutputError(secondIntermediateVector, labelVector, outputError, secondLayerRowCount);
            neuralComputeSecondLayerGradient(outputError, firstIntermediateVector, secondLayerWeightGradient, secondLayerRowCount, firstLayerRowCount);
            secondLayerBiasGradient = outputError;
          
            // Find first layer gradients
            neuralComputeFirstLayerOutputError(secondHiddenLayerWeights, outputError, firstIntermediateVector, firstLayerError, secondLayerRowCount, secondLayerColumnCount);
            neuralComputeFirstLayerGradient(firstLayerError, inputVector, firstLayerWeightGradient, firstLayerColumnCount, firstLayerRowCount);
            firstLayerBiasGradient = firstLayerError;

            // Update weights with gradients.
            neuralUpdateWeights(firstHiddenLayerWeights, firstLayerWeightGradient, LEARNING_RATE, firstLayerRowCount, firstLayerColumnCount);
            neuralUpdateWeights(secondHiddenLayerWeights, secondLayerWeightGradient, LEARNING_RATE, secondLayerRowCount, secondLayerColumnCount);
            neuralUpdateBiases(firstHiddenLayerBiases, firstLayerBiasGradient, LEARNING_RATE, firstLayerRowCount);
            neuralUpdateBiases(secondHiddenLayerBiases, secondLayerBiasGradient, LEARNING_RATE, secondLayerRowCount);
          
        }
        printf("Epoch Accuracy: %f\n", correctCount / (float)trainingImageCount);
    }
    fprintf(stderr, "Writing to file: \n");
    writeNetwork("networkFile.data", firstHiddenLayerWeights, firstHiddenLayerBiases, firstLayerColumnCount, 
        firstLayerRowCount, secondHiddenLayerWeights, secondHiddenLayerBiases, secondLayerColumnCount, secondLayerRowCount);

    printf("Running test: \n");
    int correctCount = 0;
    for(int image = 0; image < testingImageCount; image++) {
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

    printf("Freeing memory.");
    free(trainingImagesNormalized);
    free(trainingLabels);


    free(testingImagesNormalized);
    free(testingLabels);
     return 0;
}