#include "NeuralNet.h"
#include "LossOps.h"
#include <iostream>

NeuralNet::NeuralNet() 
    : inputLayer(nullptr), outputLayer(nullptr), firstLayer(nullptr) {}

NeuralNet::~NeuralNet() {
    Layer* current = inputLayer;
    while (current != nullptr) {
        Layer* next = current->getNextLayer();
        delete current;
        current = next;
    }
}

void NeuralNet::initialize(int inputSize, int inputFeatures, int outputSize, int outputFeatures) {
    inputLayer = new Layer(inputSize, inputFeatures); 
    firstLayer = inputLayer;

    // Create output layer: outputFeatures x outputSize
    outputLayer = new Layer(inputFeatures, outputSize); 
    inputLayer->setNextLayer(outputLayer);
    outputLayer->setPrevLayer(inputLayer);

    std::cout << "Neural network initialized with:\n";
    std::cout << "  Input layer: " << inputSize << " x " << inputFeatures << "\n";
    std::cout << "  Output layer: " << outputSize << " x " << outputFeatures << "\n";
}



void NeuralNet::addLayer(int inputSize, int outputSize, const std::string& activation) {
    if (outputLayer == nullptr || inputLayer == nullptr) {
        std::cerr << "Error: Neural network not initialized. Call initialize() first.\n";
        return;
    }

    // Check compatibility with the current network structure
    if (outputLayer->getInputSize() != inputSize) {
        std::cerr << "Error: Input size of new layer (" << inputSize
                  << ") does not match output size of previous layer (" 
                  << outputLayer->getInputSize() << ").\n";
        return;
    }

    Layer* newLayer = new Layer(inputSize, outputSize);

    Layer* prevLayer = outputLayer->getPrevLayer();
    if (prevLayer) {
        prevLayer->setNextLayer(newLayer);
        newLayer->setPrevLayer(prevLayer);
    }
    newLayer->setNextLayer(outputLayer);
    outputLayer->setPrevLayer(newLayer);

    std::cout << "Added hidden layer with " << inputSize << " inputs and "
              << outputSize << " outputs.\n";
}

void NeuralNet::train(const float* inputs, const float* targets, int batchSize, int numEpochs, float learningRate) {
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        float* predictions = forward(inputs, batchSize);

        float loss = computeLoss(predictions, targets, batchSize);

        backward(predictions, targets, batchSize, learningRate);

        std::cout << "Epoch " << epoch + 1 << "/" << numEpochs << ": Loss = " << loss << "\n";

        cudaFree(predictions); 
    }
}

float* NeuralNet::forward(const float* input, int batchSize) {
    const float* currentInput = input;
    Layer* currentLayer = inputLayer->getNextLayer(); 

    while (currentLayer != nullptr) {
        float* output = currentLayer->forward(currentInput, batchSize);

        if (currentInput != input) {
            cudaFree((void*)currentInput); 
        }

        currentInput = output;
        currentLayer = currentLayer->getNextLayer();
    }

    return (float*)currentInput; 
}

float NeuralNet::computeLoss(const float* predictions, const float* targets, int batchSize, int outputSize) {
    int totalElements = batchSize * outputSize;

    float loss = LossOps::MeanSquaredError(targets, predictions, totalElements);

    return loss;
}

void NeuralNet::backward(const float* predictions, const float* targets, int batchSize, float learningRate) {
    Layer* currentLayer = outputLayer;

    // Compute initial loss gradient
    float* dLoss = computeLossGradient(predictions, targets, batchSize);

    // Backpropagate through layers
    while (currentLayer != inputLayer) {
        // Compute gradients
        float* dWeights = computeWeightGradient(currentLayer, dLoss, batchSize);
        float* dBiases = computeBiasGradient(currentLayer, dLoss, batchSize);

        // Update weights and biases
        MatrixOps::scalerMultiplication(dWeights, dWeights, -learningRate, currentLayer->getInputSize(), currentLayer->getOutputSize());
        MatrixOps::add(currentLayer->getWeights(), dWeights, currentLayer->getWeights(), currentLayer->getInputSize(), currentLayer->getOutputSize());

        MatrixOps::scalerMultiplication(dBiases, dBiases, -learningRate, 1, currentLayer->getOutputSize());
        MatrixOps::add(currentLayer->getBiases(), dBiases, currentLayer->getBiases(), 1, currentLayer->getOutputSize());

        // Propagate gradients to the previous layer
        float* new_dLoss = propagateGradientBackward(currentLayer, dLoss, batchSize);
        cudaFree(dLoss);
        dLoss = new_dLoss;

        // Move to the previous layer
        currentLayer = currentLayer->getPrevLayer();

        // Free temporary memory
        cudaFree(dWeights);
        cudaFree(dBiases);
    }

    cudaFree(dLoss);
}


float* NeuralNet::computeWeightGradient(Layer* currentLayer, const float* dLoss, int batchSize) {

    int rows = currentLayer->getInputSize();
    int cols = currentLayer->getOutputSize();
    size_t size = rows * cols * sizeof(float);
    float* dWeights;
    cudaMalloc(&dWeights, size);

    // Compute weight gradients: dWeights = A_prev^T * dLoss
    const float* prevActivations = currentLayer->getPrevLayer()->forward(nullptr, batchSize); // Forward returns activations
    MatrixOps::multiply(prevActivations, dLoss, dWeights, rows, batchSize, batchSize, cols);

    return dWeights;
}

float* NeuralNet::computeBiasGradient(Layer* currentLayer, const float* dLoss, int batchSize) {
    // Allocate memory for bias gradients
    int outputSize = currentLayer->getOutputSize();
    float* dBiases;
    cudaMalloc(&dBiases, outputSize * sizeof(float));

    // Sum up gradients across the batch: dBiases = sum(dLoss, axis=0)
    MatrixOps::sumAcrossRows(dLoss, dBiases, batchSize, outputSize);

    return dBiases;
}

float* NeuralNet::propagateGradientBackward(Layer* currentLayer, const float* dLoss, int batchSize) {
    // Allocate memory for the new dLoss
    int rows = currentLayer->getInputSize();
    int cols = currentLayer->getOutputSize();
    size_t size = batchSize * rows * sizeof(float);
    float* new_dLoss;
    cudaMalloc(&new_dLoss, size);

    // Propagate gradients backward: new_dLoss = dLoss * W^T
    const float* weights = currentLayer->getWeights();
    MatrixOps::multiply(dLoss, weights, new_dLoss, batchSize, cols, cols, rows);

    return new_dLoss;
}

Layer* NeuralNet::getInputLayer() const {
    return inputLayer;
}

Layer* NeuralNet::getOutputLayer() const {
    return outputLayer;
}