#include "NeuralNet.h"
#include "LossOps.h"
#include <iostream>
#include <algorithm>
#include <random>
#include "Utils.h"

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

void NeuralNet::initialize(int inputSize, int* Nuerons, int hiddenLayers, int outputFeatures) {

    inputLayer = new Layer(1, inputSize, "relu");
    Layer* prevLayer = inputLayer;
    Layer* currentLayer;
    for(int i = 0; i < hiddenLayers; ++i) {
        currentLayer = new Layer(prevLayer->getOutputSize(), Nuerons[i], "relu");
        prevLayer->setNextLayer(currentLayer);
        currentLayer->setPrevLayer(prevLayer);
        prevLayer = currentLayer;
    }
    outputLayer = new Layer(prevLayer->getOutputSize(), outputFeatures, "softmax");
    outputLayer->setPrevLayer(prevLayer);
    prevLayer->setNextLayer(outputLayer);

}
//     int inputSize = samples * features; // Flatten input: 10x5 -> 50

//     // Initialize input layer
//     inputLayer = new Layer(inputSize, hiddenSize, "relu");
//     firstLayer = inputLayer;
    

//     // Initialize hidden layer
//     firstLayer = new Layer(hiddenSize, hiddenSize, "relu");
//     inputLayer->setNextLayer(firstLayer);
//     firstLayer->setPrevLayer(inputLayer);

//     // Initialize output layer
//     outputLayer = new Layer(hiddenSize, outputFeatures, "softmax");
//     firstLayer->setNextLayer(outputLayer);
//     outputLayer->setPrevLayer(firstLayer);

//     std::cout << "Neural network initialized with:\n";
//     std::cout << "  Flattened input size: " << inputSize << " (samples * features)\n";
//     std::cout << "  Hidden layer size: " << hiddenSize << "\n";
//     std::cout << "  Output layer size: " << outputFeatures << " (high, low prediction)\n";
// }




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

void NeuralNet::train(const float* trainingData,
                     int numDays,
                     int lookback,
                     int numFeatures,
                     int numPredictions,
                     int batchSize,
                     float learningRate,
                     int numEpochs,
                     const int* targetIndices) {
    std::cout << "Training..." << std::endl;
    int inputSize = lookback * numFeatures;
    int outputSize = numPredictions;
    int numPossibleSequences = numDays - lookback - 1;
    int numBatchesPerEpoch = numPossibleSequences / batchSize;
    
    std::vector<int> shuffledIndices = Utils::generateShuffledIndices(numPossibleSequences);
    
    // Allocate device memory for indices
    int* d_indices = nullptr;
    cudaError_t cudaStatus = cudaMalloc(&d_indices, numPossibleSequences * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return;
    }
    
    cudaStatus = cudaMemcpy(d_indices, shuffledIndices.data(), 
                           numPossibleSequences * sizeof(int), 
                           cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_indices);
        return;
    }

    std::cout << "Training Configuration:\n"
              << "  Total possible sequences: " << numPossibleSequences << "\n"
              << "  Batch size: " << batchSize << "\n"
              << "  Batches per epoch: " << numBatchesPerEpoch << "\n"
              << "  Number of epochs: " << numEpochs << "\n";
    
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        float epochLoss = 0.0f;
        
        for (int batch = 0; batch < numBatchesPerEpoch; ++batch) {
            float batchLoss = 0.0f;
            
            for (int i = 0; i < batchSize; i++) {
                int startIdx;
                cudaMemcpy(&startIdx, 
                          &d_indices[batch * batchSize + i], 
                          sizeof(int), 
                          cudaMemcpyDeviceToHost);
                
                float* sampleInput;
                cudaMalloc(&sampleInput, inputSize * sizeof(float));
                
                // Flatten and copy input sequence
                for (int day = 0; day < lookback; day++) {
                    for (int feat = 0; feat < numFeatures; feat++) {
                        int inputIdx = day * numFeatures + feat;
                        int dataIdx = (startIdx + day) * numFeatures + feat;
                        cudaMemcpy(&(inputLayer->getOutput()[inputIdx]), 
                                 &trainingData[dataIdx],
                                 sizeof(float),
                                 cudaMemcpyHostToDevice);
                    }
                }
                
                forward();
                
                // Calculate loss
                float* predictions = outputLayer->getOutput();
                float* targets;
                cudaMalloc(&targets, outputSize * sizeof(float));
                int dataIdx = (startIdx + lookback) * numFeatures;
                cudaMemcpy(targets, &trainingData[dataIdx], 
                          outputSize * sizeof(float), 
                          cudaMemcpyHostToDevice);
                
                batchLoss += LossOps::MeanSquaredError(targets, predictions, 
                                                      outputSize, true);
                
                cudaFree(sampleInput);
                cudaFree(targets);
            }
            
            epochLoss += batchLoss;
            
            if (batch < 5) {  // Print loss for first 5 batches only
                std::cout << "Batch " << batch << " loss: " << batchLoss / batchSize << std::endl;
            }
        }
        
        std::cout << "Epoch " << epoch + 1 << "/" << numEpochs 
                  << " - Average loss: " << epochLoss / (numBatchesPerEpoch * batchSize) << std::endl;
    }
    
    cudaFree(d_indices);
}



void NeuralNet::forward() {
    
    Layer* currentLayer = inputLayer;
    Layer* nextLayer = currentLayer->getNextLayer();
    float* currentOutput = currentLayer->getOutput();
    
    while(nextLayer != nullptr) {
        cudaMemset(nextLayer->getOutput(), 0, 
                  1 * nextLayer->getOutputSize() * sizeof(float));
        
        MatrixOps::multiply(currentOutput, nextLayer->getWeights(), nextLayer->getOutput(),
                           1, currentLayer->getOutputSize(), 
                           nextLayer->getInputSize(), nextLayer->getOutputSize(),
                           true);
                           
        MatrixOps::addBias(nextLayer->getOutput(), nextLayer->getBiases(), nextLayer->getOutput(),
                           1, nextLayer->getOutputSize(), true);

        currentOutput = nextLayer->getOutput();
        currentLayer = nextLayer;
        nextLayer = currentLayer->getNextLayer();
    }
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
    // Layer* currentLayer = outputLayer;

    // // Calculate the initial delta for the output layer
    // float* delta = nullptr;
    // int outputSize = currentLayer->getOutputSize();
    // cudaMalloc(&delta, batchSize * outputSize * sizeof(float));

    // // delta = 2 * (predictions - targets) / batchSize
    // MatrixOps::subtract(predictions, targets, delta, batchSize, outputSize);
    // MatrixOps::scalerMultiplication(delta, delta, 2.0f / batchSize, batchSize, outputSize);

    // // Backpropagate through each layer
    // while (currentLayer != nullptr) {
    //     int inputSize = currentLayer->getInputSize();

    //     // Compute weight gradients: dW = input^T * delta
    //     float* weightGradient;
    //     cudaMalloc(&weightGradient, inputSize * outputSize * sizeof(float));
    //     MatrixOps::transpose(currentLayer->getPrevLayerOutput(), currentLayer->getPrevLayerOutputTransposed(), batchSize, inputSize); // Transpose inputs
    //     MatrixOps::multiply(currentLayer->getPrevLayerOutputTransposed(), delta, weightGradient, inputSize, batchSize, batchSize, outputSize, true);

    //     // Compute bias gradients: db = sum(delta, axis=0)
    //     float* biasGradient;
    //     cudaMalloc(&biasGradient, outputSize * sizeof(float));
    //     MatrixOps::sumAcrossRows(delta, biasGradient, batchSize, outputSize);

    //     // Store gradients in the layer
    //     currentLayer->setWeightGradients(weightGradient);
    //     currentLayer->setBiasGradients(biasGradient);

    //     // Compute delta for the previous layer: delta_prev = delta * W^T
    //     if (currentLayer->getPrevLayer() != nullptr) {
    //         float* deltaPrev;
    //         cudaMalloc(&deltaPrev, batchSize * inputSize * sizeof(float));
    //         MatrixOps::transpose(currentLayer->getWeights(), currentLayer->getWeightsTransposed(), inputSize, outputSize); // Transpose weights
    //         MatrixOps::multiply(delta, currentLayer->getWeightsTransposed(), deltaPrev, batchSize, outputSize, outputSize, inputSize, true);

    //         cudaFree(delta); // Free the current delta
    //         delta = deltaPrev; // Update delta for the next iteration
    //     }

    //     currentLayer = currentLayer->getPrevLayer();
    // }

    // // Free the final delta memory
    // cudaFree(delta);
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
    MatrixOps::multiply(dLoss, weights, new_dLoss, batchSize, cols, cols, rows, true);

    return new_dLoss;
}

Layer* NeuralNet::getInputLayer() const {
    return inputLayer;
}

Layer* NeuralNet::getOutputLayer() const {
    return outputLayer;
}

void NeuralNet::extractBatch(const float* fullData, const std::vector<int>& indices, 
                               int startIdx, int batchSize, int numFeatures) {
    // Calculate the size of the batch in bytes
    // size_t batchSizeBytes = batchSize * numFeatures * sizeof(float);

    // // Allocate memory for the batch on the device
    // float* batchData;
    // cudaMalloc(&batchData, batchSizeBytes);
    // cudaDeviceSynchronize();

    // // Temporary host buffer for the batch
    // std::vector<float> hostBatchData(batchSize * numFeatures);

    // // Extract rows corresponding to the current batch indices
    // for (int i = 0; i < batchSize; ++i) {
    //     int rowIdx = indices[startIdx + i]; // Get the row index from shuffled indices
    //     for (int j = 0; j < numFeatures; ++j) {
    //         hostBatchData[i * numFeatures + j] = fullData[rowIdx * numFeatures + j];
    //     }
    // }

    

    // // Copy the extracted batch from host to device
    // cudaMemcpy(batchData, hostBatchData.data(), batchSizeBytes, cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();

    // return batchData; // Return device pointer to the batch
}

void NeuralNet::applyGradients(float learningRate) {
    // Layer* currentLayer = outputLayer;

    // while (currentLayer != nullptr) {
    //     // Retrieve gradients
    //     float* weightGradients = currentLayer->getWeightGradients();
    //     float* biasGradients = currentLayer->getBiasGradients();

    //     // Update weights: W = W - learningRate * dW
    //     MatrixOps::scalerMultiplication(weightGradients, weightGradients, -learningRate, currentLayer->getInputSize(), currentLayer->getOutputSize());
    //     MatrixOps::add(currentLayer->getWeights(), weightGradients, currentLayer->getWeights(), currentLayer->getInputSize(), currentLayer->getOutputSize());

    //     // Update biases: b = b - learningRate * db
    //     MatrixOps::scalerMultiplication(biasGradients, biasGradients, -learningRate, 1, currentLayer->getOutputSize());
    //     MatrixOps::add(currentLayer->getBiases(), biasGradients, currentLayer->getBiases(), 1, currentLayer->getOutputSize());

    //     currentLayer = currentLayer->getPrevLayer(); // Move to the previous layer
    // }
}

void NeuralNet::debugPrintGPUArray(const char* name, const float* d_array, int rows, int cols) {
    int totalElements = rows * cols;
    
    // Allocate host memory
    float* h_array = new float[totalElements];

    // Copy from GPU to CPU
    cudaMemcpy(h_array, d_array, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the array
    std::cout << "[DEBUG] " << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_array[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "-------------------\n";

    // Free host memory
    delete[] h_array;
}

void NeuralNet::setInput(const float* input, int rows, int cols) {
    int flattenedSize = rows * cols;
    
    // Verify input size matches the network's input layer
    if (flattenedSize != inputLayer->getOutputSize()) {
        std::cerr << "Error: Input size mismatch. Expected " << inputLayer->getOutputSize() 
                  << " but got " << flattenedSize << " (rows: " << rows 
                  << ", cols: " << cols << ")" << std::endl;
        return;
    }

    // Allocate temporary device memory for flattened input
    float* d_flattenedInput;
    cudaMalloc(&d_flattenedInput, flattenedSize * sizeof(float));

    // Copy input data to device and flatten it
    for (int i = 0; i < rows; ++i) {
        cudaMemcpy(d_flattenedInput + (i * cols), 
                  input + (i * cols), 
                  cols * sizeof(float), 
                  cudaMemcpyHostToDevice);
    }

    // Copy flattened input to input layer's output buffer
    cudaMemcpy(inputLayer->getOutput(), 
               d_flattenedInput, 
               flattenedSize * sizeof(float), 
               cudaMemcpyDeviceToDevice);

    // Free temporary device memory
    cudaFree(d_flattenedInput);

    // Debug print to verify input
    std::cout << "Set input layer with flattened data (" << rows << "x" << cols 
              << " -> 1x" << flattenedSize << "):" << std::endl;
    debugPrintGPUArray("Flattened Input", inputLayer->getOutput(), 1, flattenedSize);
}