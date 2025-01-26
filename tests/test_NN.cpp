#include "NeuralNet.h"
#include <iostream>
#include <cassert>

void testNeuralNetInitialization() {
    NeuralNet nn;

    // Initialize the network with input and output layers
    nn.initialize(5, 5, 2, 1); // Input size: 5x5, Output size: 2x1
    std::cout << "Neural network initialized successfully.\n";

    // Validate the input and output layers
    Layer* inputLayer = nn.getInputLayer();
    Layer* outputLayer = nn.getOutputLayer();
    assert(inputLayer != nullptr);
    assert(outputLayer != nullptr);

    // Check input layer dimensions
    assert(inputLayer->getInputSize() == 5);
    assert(inputLayer->getOutputSize() == 5);

    // Check output layer dimensions
    assert(outputLayer->getInputSize() == 5);
    assert(outputLayer->getOutputSize() == 2);

    std::cout << "Input and output layers validated successfully.\n";
}

void testAddHiddenLayer() {
    NeuralNet nn;

    // Initialize the network with input and output layers
    nn.initialize(5, 5, 4, 2); // Input: 5x5, Output: 4x2
    Layer* inputLayer = nn.getInputLayer();
    Layer* outputLayer = nn.getOutputLayer();

    assert(inputLayer != nullptr);
    assert(outputLayer != nullptr);

    assert(outputLayer->getInputSize() == 5);
    assert(outputLayer->getOutputSize() == 4);

    // Add a hidden layer
    nn.addLayer(5, 4, "relu");

    // Validate the hidden layer
    Layer* hiddenLayer = inputLayer->getNextLayer();
    assert(hiddenLayer != nullptr);

    // Check dimensions of the hidden layer
    assert(hiddenLayer->getInputSize() == 5);
    assert(hiddenLayer->getOutputSize() == 4);

    // Ensure connections are correct
    assert(hiddenLayer->getPrevLayer() == inputLayer);
    assert(hiddenLayer->getNextLayer() == outputLayer);
    assert(inputLayer->getNextLayer() == hiddenLayer);
    assert(outputLayer->getPrevLayer() == hiddenLayer);

    std::cout << "Hidden layer added and validated successfully.\n";
}

int main() {
    testNeuralNetInitialization();
    testAddHiddenLayer();
    std::cout << "All NeuralNet tests passed successfully!\n";
    return 0;
}
