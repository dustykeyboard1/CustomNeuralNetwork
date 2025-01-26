#include "Layer.h"
#include <iostream>

Layer::Layer(int inputSize, int outputSize, Layer* prevLayer)
    : inputSize(inputSize), outputSize(outputSize), prevLayer(prevLayer), nextLayer(nullptr) {
        size_t weightSize = inputSize * outputSize * sizeof(float);
        size_t biasSize = outputSize * sizeof(float);

        cudaMalloc(&weights, weightSize);
        cudaMalloc(&biases, biasSize);

        initualizeWeights();
        initialBiases();

        std::cout << "Initialized layer with " << inputSize << " inputs and " << outputSize << " outputs.\n";
}
         
Layer::~Layer() {
    cudaFree(weights);
    cudaFree(biases);
}

void Layer::initualizeWeights() {
    MatrixOps::initializeWeights(weights, inputSize, outputSize, "xavier");
}

void Layer::initialBiases() {
    MatrixOps::initializeWeights(biases, 1, outputSize, "uniform");
}

float* Layer::forward(const float* input, int batchSize) {
    size_t outputSizeBytes = batchSize * outputSize * sizeof(float);

    float* outputs;
    cudaMalloc(&outputs, outputSizeBytes);

    MatrixOps::multiply(input, weights, outputs, batchSize, inputSize, inputSize, outputSize);

    MatrixOps::addBias(outputs, biases, outputs, batchSize, outputSize);

    return outputs;
}

void Layer::setNextLayer(Layer* next) {
    nextLayer = next;
    if (next) {
        next -> setPrevLayer(this);
    }
}

void Layer::setPrevLayer(Layer* prev) {
    prevLayer = prev;
}

Layer* Layer::getNextLayer() const {
    return nextLayer;
}

Layer* Layer::getPrevLayer() const {
    return prevLayer;
}

int Layer::getInputSize() const {
    return inputSize;
}

int Layer::getOutputSize() const {
    return outputSize;
}