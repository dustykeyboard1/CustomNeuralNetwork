#include <iostream>
#include "NeuralNet.h"

void testNeuralNetInitialization() {
    int samples = 3;        // Number of data samples
    int features = 2;        // Number of features per sample
    int inputSize = samples * features;
    int numHiddenLayers = 2;
    int* neu = new int[numHiddenLayers]{4,8};
    int outputFeatures = 2;  // Output features

    NeuralNet nn;
    nn.initialize(inputSize, neu, numHiddenLayers, outputFeatures);
    nn.forward();
    delete[] neu;

    std::cout << "[TEST] Neural network initialization complete.\n";
}

// void testTrainInitialization() {
//     const int numSamples = 10;
//     const int numFeatures = 5;
//     const int hiddenSize = 16;
//     const int outputFeatures = 2;
//     const int batchSize = 5;
//     const int epochs = 1;
//     const float learningRate = 0.01f;

//     // Define inputs as a 10x5 feature matrix (flattened to 1D array)
//     float inputs[numSamples * numFeatures] = {
//         0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f, 1.6f, 1.8f, 2.0f,
//         2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f,
//         4.2f, 4.4f, 4.6f, 4.8f, 5.0f, 5.2f, 5.4f, 5.6f, 5.8f, 6.0f,
//         6.2f, 6.4f, 6.6f, 6.8f, 7.0f, 7.2f, 7.4f, 7.6f, 7.8f, 8.0f,
//         8.2f, 8.4f, 8.6f, 8.8f, 9.0f, 9.2f, 9.4f, 9.6f, 9.8f, 10.0f
//     };

//     // Define regression targets for 10 samples with 2 outputs per sample
//     float targets[numSamples * outputFeatures] = {
//         1.5f, 2.5f, 2.0f, 3.0f, 2.5f, 3.5f, 3.0f, 4.0f, 3.5f, 4.5f,
//         4.0f, 5.0f, 4.5f, 5.5f, 5.0f, 6.0f, 5.5f, 6.5f, 6.0f, 7.0f
//     };

//     // Initialize the neural network
//     NeuralNet nn;
//     nn.initialize(numSamples, numFeatures, hiddenSize, outputFeatures);

//     // Debug: Verify the training process
//     std::cout << "[TEST] Starting training for 1 epoch with debug statements.\n";
//     nn.train(inputs, targets, numSamples, numFeatures, batchSize, epochs, learningRate);
//     std::cout << "[TEST] Training process completed successfully.\n";
// }

int main() {
    testNeuralNetInitialization();
    // std::cout << "------------Train Test---------------" << std::endl;
    // testTrainInitialization();
    return 0;
}
