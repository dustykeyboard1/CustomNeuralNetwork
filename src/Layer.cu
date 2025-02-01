#include "Layer.h"
#include <iostream>

// Constructor: Initialize layer with dimensions and allocate GPU memory
Layer::Layer(int inputSize, int outputSize, const std::string& activationType, Layer* prevLayer)
    : inputSize(inputSize), outputSize(outputSize), activationType(activationType), 
      prevLayer(prevLayer), nextLayer(nullptr) {
    
    size_t weightSize = inputSize * outputSize * sizeof(float);
    size_t biasSize = outputSize * sizeof(float);
    size_t outputSizeBytes = 1 * outputSize * sizeof(float);

    // Allocate GPU memory for network parameters
    cudaMalloc(&weights, weightSize);
    cudaMalloc(&biases, biasSize);
    cudaMalloc(&output, outputSizeBytes);
    
    // Allocate GPU memory for gradients and intermediate values
    cudaMalloc(&weightGradients, weightSize);
    cudaMalloc(&biasGradients, biasSize);
    cudaMalloc(&delta, outputSizeBytes);
    cudaMalloc(&preActivation, outputSizeBytes);
    
    // Initialize buffers to zero
    cudaMemset(output, 0, outputSizeBytes);
    cudaMemset(weightGradients, 0, weightSize);
    cudaMemset(biasGradients, 0, biasSize);
    cudaMemset(delta, 0, outputSizeBytes);
    cudaMemset(preActivation, 0, outputSizeBytes);

    // Initialize weights and biases
    initializeWeights();
    initialBiases();

    std::cout << "Initialized layer with:\n"
              << "  - Weight matrix: " << inputSize << "x" << outputSize << "\n"
              << "  - Output size: 1x" << outputSize << "\n";
}

// Destructor: Free GPU memory
Layer::~Layer() {
    cudaFree(weights);
    cudaFree(biases);
    if (weightGradients) cudaFree(weightGradients);
    if (biasGradients) cudaFree(biasGradients);
    if (weightsTransposed) cudaFree(weightsTransposed);
    if (prevLayerOutputTransposed) cudaFree(prevLayerOutputTransposed);
}

// Weight initialization methods
void Layer::initializeWeights() {
    MatrixOps::initializeWeights(weights, inputSize, outputSize, "xavier");
}

void Layer::initialBiases() {
    MatrixOps::initializeWeights(biases, 1, outputSize, "uniform");
}

// Forward pass: Apply activation function to input
float* Layer::forward(const float* input, int batchSize) {
    // Apply appropriate activation function
    if (activationType == "relu") {
        std::cout << "[DEBUG] Applying ReLU activation...\n";
        MatrixOps::Relu(output, output, batchSize, outputSize);
    } else if (activationType == "sigmoid") {
        std::cout << "[DEBUG] Applying Sigmoid activation...\n";
        MatrixOps::Sigmoid(output, output, batchSize, outputSize);
    } else if (activationType == "softmax") {
        std::cout << "[DEBUG] Applying Softmax activation...\n";
        MatrixOps::Softmax(output, output, batchSize, outputSize);
    } else if (activationType == "tanh") {
        std::cout << "[DEBUG] Applying Tanh activation...\n";
        MatrixOps::Tanh(output, output, batchSize, outputSize);
    } else {
        std::cerr << "[ERROR] Unknown activation type: " << activationType << ". No activation applied.\n";
    }

    return output;
}

// Layer connectivity methods
void Layer::setNextLayer(Layer* next) {
    nextLayer = next;
    if (next) {
        next -> setPrevLayer(this);
    }
}

void Layer::setPrevLayer(Layer* prev) {
    prevLayer = prev;
}

// Getter methods for layer properties
Layer* Layer::getNextLayer() const { return nextLayer; }
Layer* Layer::getPrevLayer() const { return prevLayer; }
int Layer::getInputSize() const { return inputSize; }
int Layer::getOutputSize() const { return outputSize; }

// Gradient management methods
void Layer::setWeightGradients(float* gradients) {
    if (weightGradients != nullptr) {
        cudaFree(weightGradients);
    }
    weightGradients = gradients;
}

void Layer::setBiasGradients(float* gradients) {
    if (biasGradients != nullptr) {
        cudaFree(biasGradients);
    }
    biasGradients = gradients;
}

// Layer output methods
const float* Layer::getPrevLayerOutput() const {
    if (prevLayer == nullptr) {
        throw std::runtime_error("Previous layer is null. Cannot fetch output.");
    }
    return prevLayer->getOutput();
}

float* Layer::getOutput() const {
    if (output == nullptr) {
        throw std::runtime_error("Layer output is not yet computed.");
    }
    return output;
}

void Layer::setOutput(float* out) {
    if (output != nullptr) {
        cudaFree(output);
    }
    output = out;
}

// Batch size management
int Layer::getBatchSize() const { return batchSize; }
void Layer::setBatchSize(int bSize) { batchSize = bSize; }

// Getter methods for network parameters and gradients
float* Layer::getWeightGradients() const {
    if (weightGradients == nullptr) {
        throw std::runtime_error("Weight gradients have not been set.");
    }
    return weightGradients;
}

float* Layer::getBiasGradients() const {
    if (biasGradients == nullptr) {
        throw std::runtime_error("Bias gradients have not been set.");
    }
    return biasGradients;
}

float* Layer::getWeights() const { return weights; }
float* Layer::getBiases() const { return biases; }
std::string Layer::getActivationType() const { return activationType; }
float* Layer::getDelta() { return delta; }
float* Layer::getPreActivation() { return preActivation; }

