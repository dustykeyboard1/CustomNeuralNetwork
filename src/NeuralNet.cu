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
    // Add normalization at the start
    std::cout << "Normalizing input data...\n";
    
    // Find max values for each feature
    std::vector<float> maxValues(numFeatures, std::numeric_limits<float>::min());
    for (int i = 0; i < numDays; i++) {
        for (int j = 0; j < numFeatures; j++) {
            maxValues[j] = std::max(maxValues[j], std::abs(trainingData[i * numFeatures + j]));
        }
    }
    
    // Create normalized copy of training data
    float* normalizedData;
    cudaMallocHost(&normalizedData, numDays * numFeatures * sizeof(float));
    for (int i = 0; i < numDays; i++) {
        for (int j = 0; j < numFeatures; j++) {
            normalizedData[i * numFeatures + j] = trainingData[i * numFeatures + j] / maxValues[j];
        }
    }
    
    std::cout << "Feature normalization factors:\n";
    for (int i = 0; i < numFeatures; i++) {
        std::cout << "Feature " << i << ": " << maxValues[i] << "\n";
    }
    
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

    std::cout << "\nTraining Configuration:\n"
              << "  Total possible sequences: " << numPossibleSequences << "\n"
              << "  Batch size: " << batchSize << "\n"
              << "  Batches per epoch: " << numBatchesPerEpoch << "\n"
              << "  Number of epochs: " << numEpochs << "\n"
              << "  Learning rate: " << learningRate << "\n\n";
    
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        std::cout << "\nEpoch " << epoch + 1 << "/" << numEpochs << std::endl;
        float epochLoss = 0.0f;
        
        for (int batch = 0; batch < numBatchesPerEpoch; ++batch) {
            float batchLoss = 0.0f;
            bool isFirstBatch = (batch == 0);
            
            // Reset static flags for new batch
            if (isFirstBatch) {
                std::cout << "\nProcessing first batch:" << std::endl;
            }
            
            // Reset accumulated gradients at the start of each batch
            Layer* layer = inputLayer->getNextLayer();
            while (layer != nullptr) {
                

                cudaMemset(layer->getWeightGradients(), 0, 
                                  layer->getInputSize() * layer->getOutputSize() * sizeof(float));
                
                
                cudaMemset(layer->getBiasGradients(), 0, 
                                  layer->getOutputSize() * sizeof(float));
                
                layer = layer->getNextLayer();
            }
            // Process each sample in the batch
            for (int i = 0; i < batchSize; i++) {
                int startIdx;
                cudaMemcpy(&startIdx, &d_indices[batch * batchSize + i], sizeof(int), cudaMemcpyDeviceToHost);
                
                // Load input data
                for (int day = 0; day < lookback; day++) {
                    for (int feat = 0; feat < numFeatures; feat++) {
                        cudaMemcpy(&(inputLayer->getOutput()[day * numFeatures + feat]), 
                                 &normalizedData[(startIdx + day) * numFeatures + feat],
                                 sizeof(float), cudaMemcpyHostToDevice);
                    }
                }

                // Debug print for first sample in batch
                if (i == 0) {
                    std::cout << "\nBatch " << batch + 1 << ", First Sample:" << std::endl;
                    
                    // Print input
                    float* h_input = new float[lookback * numFeatures];
                    cudaMemcpy(h_input, inputLayer->getOutput(), 
                             lookback * numFeatures * sizeof(float), 
                             cudaMemcpyDeviceToHost);
                    std::cout << "Input: ";
                    for (int j = 0; j < lookback * numFeatures; j++) {
                        std::cout << h_input[j] << " ";
                    }
                    std::cout << std::endl;
                    delete[] h_input;
                }

                forward();
                
                // Get target and calculate loss
                float* targets;
                cudaMalloc(&targets, outputSize * sizeof(float));
                cudaMemcpy(targets, &normalizedData[(startIdx + lookback) * numFeatures], 
                          outputSize * sizeof(float), cudaMemcpyHostToDevice);

                // Debug print for first sample in batch
                if (i == 0) {
                    // Print network output
                    float* h_output = new float[outputSize];
                    cudaMemcpy(h_output, outputLayer->getOutput(), 
                             outputSize * sizeof(float), 
                             cudaMemcpyDeviceToHost);
                    std::cout << "Network Output: ";
                    for (int j = 0; j < outputSize; j++) {
                        std::cout << h_output[j] << " ";
                    }
                    std::cout << std::endl;
                    delete[] h_output;

                    // Print target
                    float* h_target = new float[outputSize];
                    cudaMemcpy(h_target, targets, 
                             outputSize * sizeof(float), 
                             cudaMemcpyDeviceToHost);
                    std::cout << "Target: ";
                    for (int j = 0; j < outputSize; j++) {
                        std::cout << h_target[j] << " ";
                    }
                    std::cout << std::endl;
                    delete[] h_target;
                }
                
                float sampleLoss = LossOps::MeanSquaredError(targets, outputLayer->getOutput(), 
                                                           outputSize, false);
                batchLoss += sampleLoss;
                
                backward(outputLayer->getOutput(), targets, 1);
                cudaFree(targets);
            }
            
            // Average the accumulated gradients
            layer = inputLayer->getNextLayer();
            while (layer != nullptr) {
                int inputSize = layer->getInputSize();
                int outputSize = layer->getOutputSize();
                
                // Average weight gradients
                MatrixOps::scalerMultiplication(layer->getWeightGradients(), 
                                              layer->getWeightGradients(),
                                              1.0f/batchSize,
                                              inputSize, outputSize, true);
                
                // Average bias gradients
                MatrixOps::scalerMultiplication(layer->getBiasGradients(), 
                                              layer->getBiasGradients(),
                                              1.0f/batchSize,
                                              1, outputSize, true);
                
                layer = layer->getNextLayer();
            }
            
            // Apply averaged gradients
            applyGradients(learningRate);
            
            // Average batch loss
            batchLoss /= batchSize;
            epochLoss += batchLoss;
            
            if ((batch + 1) % 10 == 0 || batch == 0) {
                std::cout << "Batch " << batch + 1 << "/" << numBatchesPerEpoch 
                         << " - Loss: " << batchLoss/batchSize << std::endl;
            }
        }
        
        epochLoss /= numBatchesPerEpoch;
        std::cout << "Epoch " << epoch + 1 << " average loss: " << epochLoss << std::endl;
    }
    
    cudaFree(d_indices);
    cudaFreeHost(normalizedData);
}



void NeuralNet::forward() {
    Layer* currentLayer = inputLayer;
    Layer* nextLayer = currentLayer->getNextLayer();
    float* currentOutput = currentLayer->getOutput();
    int layerIndex = 0;
    
    // Only print for first sample of batch
    static bool firstSample = true;
    
    while(nextLayer != nullptr) {
        cudaMemset(nextLayer->getOutput(), 0, 
                  1 * nextLayer->getOutputSize() * sizeof(float));
        
        MatrixOps::multiply(currentOutput, nextLayer->getWeights(), nextLayer->getOutput(),
                           1, currentLayer->getOutputSize(), 
                           nextLayer->getInputSize(), nextLayer->getOutputSize(),
                           true);
                           
        MatrixOps::addBias(nextLayer->getOutput(), nextLayer->getBiases(), nextLayer->getOutput(),
                           1, nextLayer->getOutputSize(), true);
        
        if (nextLayer != outputLayer) {
            applyActivation(nextLayer);
        }

        if (firstSample) {
            float* h_output = new float[nextLayer->getOutputSize()];
            cudaMemcpy(h_output, nextLayer->getOutput(), 
                      nextLayer->getOutputSize() * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            std::cout << "Layer " << layerIndex << " output: ";
            for (int i = 0; i < nextLayer->getOutputSize(); i++) {
                std::cout << h_output[i] << " ";
            }
            std::cout << std::endl;
            delete[] h_output;
        }

        currentOutput = nextLayer->getOutput();
        currentLayer = nextLayer;
        nextLayer = currentLayer->getNextLayer();
        layerIndex++;
    }
    firstSample = false;
}

void NeuralNet::applyActivation(Layer* currentLayer) {
    if (currentLayer->getActivationType() == "relu") {
        MatrixOps::Relu(currentLayer->getOutput(), currentLayer->getOutput(), 
                              1, currentLayer->getOutputSize(), true);
    } else if (currentLayer->getActivationType() == "softmax") {
        MatrixOps::Softmax(currentLayer->getOutput(), currentLayer->getOutput(), 
                              1, currentLayer->getOutputSize(), true);
    } else if (currentLayer->getActivationType() == "sigmoid") {
        MatrixOps::Sigmoid(currentLayer->getOutput(), currentLayer->getOutput(), 
                              1, currentLayer->getOutputSize(), true);
    } else if (currentLayer->getActivationType() == "tanh") {
        MatrixOps::Tanh(currentLayer->getOutput(), currentLayer->getOutput(), 
                              1, currentLayer->getOutputSize(), true);
    }
}



float NeuralNet::calculateLoss(const float* predictions, const float* targets, int batchSize) {
    int outputSize = outputLayer->getOutputSize();
    int size = batchSize * outputSize;
    return LossOps::MeanSquaredError(targets, predictions, size);
}


void NeuralNet::backward(const float* predictions, const float* targets, int batchSize) {
    static bool firstSample = true;
    if (firstSample) {
        std::cout << "\nBackward Pass Details:" << std::endl;
        float* h_pred = new float[outputLayer->getOutputSize()];
        float* h_targ = new float[outputLayer->getOutputSize()];
        cudaMemcpy(h_pred, predictions, outputLayer->getOutputSize() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_targ, targets, outputLayer->getOutputSize() * sizeof(float), cudaMemcpyDeviceToHost);
        
        std::cout << "Error calculation:" << std::endl;
        for (int i = 0; i < outputLayer->getOutputSize(); i++) {
            std::cout << "  Output " << i << ":" << std::endl;
            std::cout << "    Prediction: " << h_pred[i] << std::endl;
            std::cout << "    Target: " << h_targ[i] << std::endl;
            std::cout << "    Error: " << h_pred[i] - h_targ[i] << std::endl;
        }
        delete[] h_pred;
        delete[] h_targ;
    }

    // Calculate output gradient (2 * error for MSE loss)
    float* outputGradient;
    cudaMalloc(&outputGradient, outputLayer->getOutputSize() * sizeof(float));
    MatrixOps::subtract(predictions, targets, outputGradient, 
                       1, outputLayer->getOutputSize(), true);
    MatrixOps::scalerMultiplication(outputGradient, outputGradient, 
                                   2.0f, 1, outputLayer->getOutputSize(), true);

    // Propagate gradients through the network
    Layer* currentLayer = outputLayer;
    float* currentGradient = outputGradient;
    
    while (currentLayer != inputLayer) {
        Layer* prevLayer = currentLayer->getPrevLayer();
        
        // Scale down gradients if they're too large
        float* h_grad = new float[currentLayer->getOutputSize()];
        cudaMemcpy(h_grad, currentGradient, 
                  currentLayer->getOutputSize() * sizeof(float), 
                  cudaMemcpyDeviceToHost);
        
        float maxGrad = 0.0f;
        for (int i = 0; i < currentLayer->getOutputSize(); i++) {
            maxGrad = std::max(maxGrad, std::abs(h_grad[i]));
        }
        
        if (maxGrad > 1.0f) {
            float scale = 1.0f / maxGrad;
            MatrixOps::scalerMultiplication(currentGradient, currentGradient, 
                                          scale, 1, currentLayer->getOutputSize(), true);
        }
        delete[] h_grad;
        
        // Compute weight gradients
        MatrixOps::multiply(prevLayer->getOutput(), currentGradient,
                          currentLayer->getWeightGradients(),
                          1, prevLayer->getOutputSize(),
                          prevLayer->getOutputSize(), currentLayer->getOutputSize(),
                          true);
        
        // Compute bias gradients
        cudaMemcpy(currentLayer->getBiasGradients(), currentGradient,
                  currentLayer->getOutputSize() * sizeof(float),
                  cudaMemcpyDeviceToDevice);
        
        if (firstSample) {
            float* h_weightGrad = new float[5];  // Just show first 5
            cudaMemcpy(h_weightGrad, currentLayer->getWeightGradients(),
                      5 * sizeof(float), cudaMemcpyDeviceToHost);
            
            std::cout << "\nLayer weight gradients:" << std::endl;
            std::cout << "  First 5 gradients: ";
            for (int i = 0; i < 5; i++) {
                std::cout << h_weightGrad[i] << " ";
            }
            std::cout << std::endl;
            
            delete[] h_weightGrad;
        }
        
        // Compute next gradient for backprop
        float* nextGradient;
        cudaMalloc(&nextGradient, prevLayer->getOutputSize() * sizeof(float));
        MatrixOps::multiply(currentGradient, currentLayer->getWeights(),
                          nextGradient,
                          1, currentLayer->getOutputSize(),
                          currentLayer->getOutputSize(), prevLayer->getOutputSize(),
                          true);
        
        if (currentGradient != outputGradient) {
            cudaFree(currentGradient);
        }
        currentGradient = nextGradient;
        currentLayer = prevLayer;
    }

    if (firstSample) {
        firstSample = false;
    }
    
    cudaFree(outputGradient);
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
    static bool firstBatch = true;
    Layer* currentLayer = outputLayer;
    int layerIdx = 0;
    
    while (currentLayer != inputLayer) {
        if (firstBatch) {
            float* h_weights = new float[currentLayer->getInputSize() * currentLayer->getOutputSize()];
            cudaMemcpy(h_weights, currentLayer->getWeights(), 
                      currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            
            std::cout << "Layer " << layerIdx << " weight update:" << std::endl;
            std::cout << "  Before max: " << *std::max_element(h_weights, 
                                        h_weights + currentLayer->getInputSize() * currentLayer->getOutputSize()) << std::endl;
            
            // Apply gradients
            MatrixOps::scalerMultiplication(currentLayer->getWeightGradients(), 
                                          currentLayer->getWeightGradients(),
                                          -learningRate,
                                          currentLayer->getInputSize(), 
                                          currentLayer->getOutputSize(), 
                                          true);
            
            MatrixOps::add(currentLayer->getWeights(), 
                          currentLayer->getWeightGradients(), 
                          currentLayer->getWeights(),
                          currentLayer->getInputSize(), 
                          currentLayer->getOutputSize(), 
                          true);
            
            cudaMemcpy(h_weights, currentLayer->getWeights(), 
                      currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            
            std::cout << "  After max: " << *std::max_element(h_weights, 
                                       h_weights + currentLayer->getInputSize() * currentLayer->getOutputSize()) << std::endl;
            
            delete[] h_weights;
        }
        
        currentLayer = currentLayer->getPrevLayer();
        layerIdx++;
    }
    firstBatch = false;
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