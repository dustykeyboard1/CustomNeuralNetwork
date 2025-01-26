#include "NeuralNet.h"
#include "LossOps.h"
#include <iostream>
#include <algorithm>
#include <random>

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

void NeuralNet::initialize(int samples, int features, int hiddenSize, int outputFeatures) {
    int inputSize = samples * features; // Flatten input: 10x5 -> 50

    // Initialize input layer
    inputLayer = new Layer(inputSize, hiddenSize, "relu");
    firstLayer = inputLayer;
    

    // Initialize hidden layer
    firstLayer = new Layer(hiddenSize, hiddenSize, "relu");
    inputLayer->setNextLayer(firstLayer);
    firstLayer->setPrevLayer(inputLayer);

    // Initialize output layer
    outputLayer = new Layer(hiddenSize, outputFeatures, "softmax");
    firstLayer->setNextLayer(outputLayer);
    outputLayer->setPrevLayer(firstLayer);

    std::cout << "Neural network initialized with:\n";
    std::cout << "  Flattened input size: " << inputSize << " (samples * features)\n";
    std::cout << "  Hidden layer size: " << hiddenSize << "\n";
    std::cout << "  Output layer size: " << outputFeatures << " (high, low prediction)\n";
}




// void NeuralNet::addLayer(int inputSize, int outputSize, const std::string& activation) {
//     if (outputLayer == nullptr || inputLayer == nullptr) {
//         std::cerr << "Error: Neural network not initialized. Call initialize() first.\n";
//         return;
//     }

//     // Check compatibility with the current network structure
//     if (outputLayer->getInputSize() != inputSize) {
//         std::cerr << "Error: Input size of new layer (" << inputSize
//                   << ") does not match output size of previous layer (" 
//                   << outputLayer->getInputSize() << ").\n";
//         return;
//     }

//     Layer* newLayer = new Layer(inputSize, outputSize);

//     Layer* prevLayer = outputLayer->getPrevLayer();
//     if (prevLayer) {
//         prevLayer->setNextLayer(newLayer);
//         newLayer->setPrevLayer(prevLayer);
//     }
//     newLayer->setNextLayer(outputLayer);
//     outputLayer->setPrevLayer(newLayer);

//     std::cout << "Added hidden layer with " << inputSize << " inputs and "
//               << outputSize << " outputs.\n";
// }

void NeuralNet::train(const float* inputs, const float* targets, int numSamples, int numFeatures, int batchSize, int epochs, float learningRate) {
    // Helper variables
    int numBatches = (numSamples + batchSize - 1) / batchSize; // Ceiling of numSamples / batchSize
    std::vector<int> indices(numSamples);

    // Initialize indices for shuffling
    for (int i = 0; i < numSamples; ++i) {
        indices[i] = i;
    }

    std::cout << "[DEBUG] Starting training with " << epochs << " epochs, batch size: " << batchSize << ", and learning rate: " << learningRate << "\n";

    // Epoch loop
    std::random_device rd; // Seed
    std::default_random_engine rng(rd());
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle data
        std::shuffle(indices.begin(), indices.end(), rng);
        std::cout << "[DEBUG] Epoch " << epoch + 1 << "/" << epochs << ": Data shuffled\n";

        float totalLoss = 0.0f;

        // Batch loop
        for (int batch = 0; batch < numBatches; ++batch) {
            std::cout << "[DEBUG] Processing batch " << batch + 1 << "/" << numBatches << "\n";

            // Extract current batch data
            int startIdx = batch * batchSize;
            int endIdx = std::min(startIdx + batchSize, numSamples);
            int currentBatchSize = endIdx - startIdx;

            std::cout << "[DEBUG] Extracting batch data: startIdx = " << startIdx << ", endIdx = " << endIdx << "\n";
            float* batchInputs = extractBatch(inputs, indices, startIdx, currentBatchSize, numFeatures);
            float* batchTargets = extractBatch(targets, indices, startIdx, currentBatchSize, 2); // 2 output features

            // Forward pass
            std::cout << "[DEBUG] Starting forward pass for batch " << batch + 1 << "\n";
            float* predictions = forward(batchInputs, currentBatchSize);

            // Debugging: Print predictions and targets
            std::cout << "[DEBUG] Batch " << batch + 1 << " predictions (first few):\n";
            float hostPredictions[10]; // Limit to first 10 values for readability
            cudaMemcpy(hostPredictions, predictions, std::min(10, currentBatchSize * 2) * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < std::min(10, currentBatchSize * 2); ++i) {
                std::cout << hostPredictions[i] << " ";
            }
            std::cout << "\n";

            std::cout << "[DEBUG] Batch " << batch + 1 << " targets (first few):\n";
            float hostTargets[10]; // Limit to first 10 values for readability
            cudaMemcpy(hostTargets, batchTargets, std::min(10, currentBatchSize * 2) * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < std::min(10, currentBatchSize * 2); ++i) {
                std::cout << hostTargets[i] << " ";
            }
            std::cout << "\n";

            // Loss calculation
            std::cout << "[DEBUG] Calculating loss for batch " << batch + 1 << "\n";
            float batchLoss = calculateLoss(predictions, batchTargets, currentBatchSize);
            std::cout << "[DEBUG] Batch " << batch + 1 << " loss: " << batchLoss << "\n";
            totalLoss += batchLoss;

            // Backward pass
            std::cout << "[DEBUG] Starting backward pass for batch " << batch + 1 << "\n";
            backward(predictions, batchTargets, currentBatchSize);

            // Update weights
            std::cout << "[DEBUG] Applying gradients for batch " << batch + 1 << "\n";
            applyGradients(learningRate);

            // Free memory for batch inputs and targets
            std::cout << "[DEBUG] Freeing memory for batch " << batch + 1 << "\n";
            cudaFree(batchInputs);
            cudaFree(batchTargets);
            cudaFree(predictions);
        }

        // Log progress for the epoch
        std::cout << "[DEBUG] Epoch " << epoch + 1 << " completed. Average Loss: " << totalLoss / numBatches << "\n";
    }

    std::cout << "[DEBUG] Training completed.\n";
}



float* NeuralNet::forward(const float* input, int batchSize) {
    std::cout << "[DEBUG] Starting forward pass...\n";

    // Debug: Initial input to the network
    std::cout << "[DEBUG] Input to the network (first 5 values): ";
    float hostInput[5];
    cudaMemcpy(hostInput, input, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < 5; ++i) {
        std::cout << hostInput[i] << " ";
    }
    std::cout << "\n";

    const float* currentInput = input;
    Layer* currentLayer = inputLayer;
    float* output = nullptr;

    int layerIndex = 0;

    // Debug: Layer-wise iteration
    while (currentLayer) {
        std::cout << "[DEBUG] Processing Layer " << layerIndex 
                  << " | Input Size: " << currentLayer->getInputSize() 
                  << " | Output Size: " << currentLayer->getOutputSize() << "\n";

        // Debug: Current input to the layer
        std::cout << "[DEBUG] Input to Layer " << layerIndex << " (first 5 values): ";
        cudaMemcpy(hostInput, currentInput, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int i = 0; i < 5; ++i) {
            std::cout << hostInput[i] << " ";
        }
        std::cout << "\n";

        // Perform the forward pass for the current layer
        output = currentLayer->forward(currentInput, batchSize);

        // Debug: Output of the layer
        std::cout << "[DEBUG] Output of Layer " << layerIndex << " (first 5 values): ";
        float hostOutput[5];
        cudaMemcpy(hostOutput, output, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int i = 0; i < 5; ++i) {
            std::cout << hostOutput[i] << " ";
        }
        std::cout << "\n";

        // Prepare for the next layer
        currentInput = output;
        currentLayer = currentLayer->getNextLayer();
        ++layerIndex;
    }

    // Debug: Final output of the network
    std::cout << "[DEBUG] Final output of the network (first 5 values): ";
    float finalOutput[5];
    cudaMemcpy(finalOutput, output, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < 5; ++i) {
        std::cout << finalOutput[i] << " ";
    }
    std::cout << "\n";

    std::cout << "[DEBUG] Forward pass completed.\n";

    return output;
}


float NeuralNet::calculateLoss(const float* predictions, const float* targets, int batchSize) {
    // Calculate the total number of elements in the batch
    int outputSize = outputLayer->getOutputSize();
    int size = batchSize * outputSize;

    // Use the provided `LossOps::MeanSquaredError` function
    float loss = LossOps::MeanSquaredError(targets, predictions, size);

    return loss; // Return the average loss
}


void NeuralNet::backward(const float* predictions, const float* targets, int batchSize) {
    Layer* currentLayer = outputLayer;

    // Calculate the initial delta for the output layer
    float* delta = nullptr;
    int outputSize = currentLayer->getOutputSize();
    cudaMalloc(&delta, batchSize * outputSize * sizeof(float));

    // delta = 2 * (predictions - targets) / batchSize
    MatrixOps::subtract(predictions, targets, delta, batchSize, outputSize);
    MatrixOps::scalerMultiplication(delta, delta, 2.0f / batchSize, batchSize, outputSize);

    // Backpropagate through each layer
    while (currentLayer != nullptr) {
        int inputSize = currentLayer->getInputSize();

        // Compute weight gradients: dW = input^T * delta
        float* weightGradient;
        cudaMalloc(&weightGradient, inputSize * outputSize * sizeof(float));
        MatrixOps::transpose(currentLayer->getPrevLayerOutput(), currentLayer->getPrevLayerOutputTransposed(), batchSize, inputSize); // Transpose inputs
        MatrixOps::multiply(currentLayer->getPrevLayerOutputTransposed(), delta, weightGradient, inputSize, batchSize, batchSize, outputSize);

        // Compute bias gradients: db = sum(delta, axis=0)
        float* biasGradient;
        cudaMalloc(&biasGradient, outputSize * sizeof(float));
        MatrixOps::sumAcrossRows(delta, biasGradient, batchSize, outputSize);

        // Store gradients in the layer
        currentLayer->setWeightGradients(weightGradient);
        currentLayer->setBiasGradients(biasGradient);

        // Compute delta for the previous layer: delta_prev = delta * W^T
        if (currentLayer->getPrevLayer() != nullptr) {
            float* deltaPrev;
            cudaMalloc(&deltaPrev, batchSize * inputSize * sizeof(float));
            MatrixOps::transpose(currentLayer->getWeights(), currentLayer->getWeightsTransposed(), inputSize, outputSize); // Transpose weights
            MatrixOps::multiply(delta, currentLayer->getWeightsTransposed(), deltaPrev, batchSize, outputSize, outputSize, inputSize);

            cudaFree(delta); // Free the current delta
            delta = deltaPrev; // Update delta for the next iteration
        }

        currentLayer = currentLayer->getPrevLayer();
    }

    // Free the final delta memory
    cudaFree(delta);
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

float* NeuralNet::extractBatch(const float* fullData, const std::vector<int>& indices, 
                               int startIdx, int batchSize, int numFeatures) {
    // Calculate the size of the batch in bytes
    size_t batchSizeBytes = batchSize * numFeatures * sizeof(float);

    // Allocate memory for the batch on the device
    float* batchData;
    cudaMalloc(&batchData, batchSizeBytes);
    cudaDeviceSynchronize();

    // Temporary host buffer for the batch
    std::vector<float> hostBatchData(batchSize * numFeatures);

    // Extract rows corresponding to the current batch indices
    for (int i = 0; i < batchSize; ++i) {
        int rowIdx = indices[startIdx + i]; // Get the row index from shuffled indices
        for (int j = 0; j < numFeatures; ++j) {
            hostBatchData[i * numFeatures + j] = fullData[rowIdx * numFeatures + j];
        }
    }

    

    // Copy the extracted batch from host to device
    cudaMemcpy(batchData, hostBatchData.data(), batchSizeBytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    return batchData; // Return device pointer to the batch
}

void NeuralNet::applyGradients(float learningRate) {
    Layer* currentLayer = outputLayer;

    while (currentLayer != nullptr) {
        // Retrieve gradients
        float* weightGradients = currentLayer->getWeightGradients();
        float* biasGradients = currentLayer->getBiasGradients();

        // Update weights: W = W - learningRate * dW
        MatrixOps::scalerMultiplication(weightGradients, weightGradients, -learningRate, currentLayer->getInputSize(), currentLayer->getOutputSize());
        MatrixOps::add(currentLayer->getWeights(), weightGradients, currentLayer->getWeights(), currentLayer->getInputSize(), currentLayer->getOutputSize());

        // Update biases: b = b - learningRate * db
        MatrixOps::scalerMultiplication(biasGradients, biasGradients, -learningRate, 1, currentLayer->getOutputSize());
        MatrixOps::add(currentLayer->getBiases(), biasGradients, currentLayer->getBiases(), 1, currentLayer->getOutputSize());

        currentLayer = currentLayer->getPrevLayer(); // Move to the previous layer
    }
}
