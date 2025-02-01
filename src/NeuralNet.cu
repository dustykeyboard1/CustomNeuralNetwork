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

    float* normalizedData;
    cudaMallocHost(&normalizedData, numDays * numFeatures * sizeof(float));
    
    int outputSize = numPredictions;
    int numPossibleSequences = numDays - lookback - 1;
    int numBatchesPerEpoch = numPossibleSequences / batchSize;
    
    std::vector<int> shuffledIndices = Utils::generateShuffledIndices(numPossibleSequences);
    
    int* d_indices = nullptr;
    cudaMalloc(&d_indices, numPossibleSequences * sizeof(int));
    cudaMemcpy(d_indices, shuffledIndices.data(), numPossibleSequences * sizeof(int), cudaMemcpyHostToDevice);

    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        for (int batch = 0; batch < numBatchesPerEpoch; ++batch) {
            Layer* layer = inputLayer->getNextLayer();
            while (layer != nullptr) {
                cudaMemset(layer->getWeightGradients(), 0, layer->getInputSize() * layer->getOutputSize() * sizeof(float));
                cudaMemset(layer->getBiasGradients(), 0, layer->getOutputSize() * sizeof(float));
                layer = layer->getNextLayer();
            }

            for (int i = 0; i < batchSize; i++) {
                int startIdx;
                cudaMemcpy(&startIdx, &d_indices[batch * batchSize + i], sizeof(int), cudaMemcpyDeviceToHost);
                
                // Calculate size of input sequence (lookback days * features)
                size_t sequenceSize = lookback * numFeatures * sizeof(float);
                
                // Copy entire sequence at once
                cudaMemcpy(inputLayer->getOutput(),
                          &trainingData[startIdx * numFeatures],  // Start of sequence
                          sequenceSize,
                          cudaMemcpyHostToDevice);

                forward();
                
                float* targets;
                cudaMalloc(&targets, outputSize * sizeof(float));
                cudaMemcpy(targets, &trainingData[(startIdx + lookback) * numFeatures], 
                          outputSize * sizeof(float), cudaMemcpyHostToDevice);
                
                backward(outputLayer->getOutput(), targets, 1);
                cudaFree(targets);
            }
            
            applyGradients(learningRate, batchSize);
        }
    }
    
    cudaFree(d_indices);

}



void NeuralNet::forward() {
    Layer* currentLayer = inputLayer->getNextLayer();
    int layerIndex = 0;
    
    // Only print for first sample of batch
    static bool firstSample = true;
    Layer* prevLayer = nullptr;
    while(currentLayer->getNextLayer() != nullptr) {
        prevLayer = currentLayer->getPrevLayer();
        cudaMemset(currentLayer->getOutput(), 0, 
                  1 * currentLayer->getOutputSize() * sizeof(float));
        
        MatrixOps::multiply(prevLayer->getOutput(), currentLayer->getWeights(), currentLayer->getOutput(),
                           prevLayer->getInputSize(), prevLayer->getOutputSize(), 
                           currentLayer->getInputSize(), currentLayer->getOutputSize(),
                           true);

        MatrixOps::addBias(prevLayer->getOutput(), currentLayer->getBiases(), currentLayer->getOutput(),
                           1, currentLayer->getOutputSize(), true);

        if (currentLayer->getNextLayer() != outputLayer) {
            cudaMemcpy(currentLayer->getPreActivation(), currentLayer->getOutput(), 
                      currentLayer->getOutputSize() * sizeof(float), 
                      cudaMemcpyDeviceToDevice);
            applyActivation(currentLayer);
        }




        if (firstSample) {
            float* h_output = new float[currentLayer->getOutputSize()];
            cudaMemcpy(h_output, currentLayer->getOutput(), 
                      currentLayer->getOutputSize() * sizeof(float), 
                      cudaMemcpyDeviceToHost);
            std::cout << "Layer " << layerIndex << " output: ";
            for (int i = 0; i < currentLayer->getOutputSize(); i++) {
                std::cout << h_output[i] << " ";
            }
            std::cout << std::endl;
            delete[] h_output;
        }

        
        currentLayer = currentLayer->getNextLayer();
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

void NeuralNet::applyAntiActivation(Layer* currentLayer) {
    if (currentLayer->getActivationType() == "relu") {
        MatrixOps::ReluGradient(currentLayer->getPreActivation(), currentLayer->getPreActivation(),
                                 1, currentLayer->getOutputSize(), true);
    } else if (currentLayer->getActivationType() == "sigmoid") {
        MatrixOps::SigmoidGradient(currentLayer->getPreActivation(), currentLayer->getPreActivation(),
                                    1, currentLayer->getOutputSize(), true);
    } else if (currentLayer->getActivationType() == "tanh") {
        MatrixOps::TanhGradient(currentLayer->getPreActivation(), currentLayer->getPreActivation(),
                                 1, currentLayer->getOutputSize(), true);
    } else if (currentLayer->getActivationType() == "softmax") {
        // Softmax derivative is handled differently during loss calculation
        // No explicit derivative calculation needed here
        return;
    }

}




float NeuralNet::calculateLoss(const float* predictions, const float* targets, int batchSize) {
    int outputSize = outputLayer->getOutputSize();
    int size = batchSize * outputSize;
    return LossOps::MeanSquaredError(targets, predictions, size);
}


void NeuralNet::backward(const float* predictions, const float* targets, int batchSize) {
    Layer* currentLayer = outputLayer;
    float* dlossW = nullptr;
    float* wT;
    float* aT;
    float* wT_current;
    
    while (currentLayer != inputLayer) {
        Layer* prevLayer = currentLayer->getPrevLayer();
        float* delta;
        cudaMalloc(&delta, currentLayer->getOutputSize() * sizeof(float));
        cudaMemset(delta, 0, currentLayer->getOutputSize() * sizeof(float));

        if (currentLayer == outputLayer) {
            std::cout << "Output layer" << std::endl;
            LossOps::computeError(targets, predictions, currentLayer->getDelta(), currentLayer->getOutputSize(), true);
            applyAntiActivation(currentLayer);
            MatrixOps::elementWiseMultiply(currentLayer->getPreActivation(), currentLayer->getDelta(), currentLayer->getDelta(), 
                                            1, currentLayer->getOutputSize(), true);

            std::cout << "point a" << std::endl;
            cudaMalloc(&aT, currentLayer->getNextLayer()->getOutputSize() * sizeof(float));
            std::cout << "point a.1" << std::endl;
            cudaMemcpy(aT, currentLayer->getNextLayer()->getPreActivation(), currentLayer->getNextLayer()->getOutputSize() * sizeof(float), cudaMemcpyDeviceToDevice);
            std::cout << "point a.2" << std::endl;


            cudaMalloc(&dlossW, currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float));
            cudaMemset(dlossW, 0, currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float));
            std::cout << "point b" << std::endl;
            MatrixOps::multiply(aT, currentLayer->getDelta(), dlossW, 
                                    currentLayer->getPrevLayer()->getOutputSize(),1,
                                    1, currentLayer->getOutputSize(), true);
            std::cout << "point c" << std::endl;
            MatrixOps::add(currentLayer->getWeightGradients(), dlossW, currentLayer->getWeightGradients(), 
                           currentLayer->getInputSize(), currentLayer->getOutputSize(), true);
            std::cout << "point d" << std::endl;

            MatrixOps::add(currentLayer->getBiasGradients(), currentLayer->getDelta(), currentLayer->getBiasGradients(), 
                          1, currentLayer->getOutputSize(), true);
            std::cout << "point e" << std::endl;

            cudaFree(dlossW);
            cudaFree(aT);
            cudaFree(delta);

        } else {
            std::cout << "Hidden layer" << std::endl;
            cudaMalloc(&wT, currentLayer->getNextLayer()->getInputSize() * currentLayer->getNextLayer()->getOutputSize() * sizeof(float));
            cudaMemcpy(wT, currentLayer->getNextLayer()->getWeights(), currentLayer->getNextLayer()->getInputSize() * currentLayer->getNextLayer()->getOutputSize() * sizeof(float), cudaMemcpyDeviceToDevice);
            MatrixOps::transpose(wT, currentLayer->getNextLayer()->getWeights(), currentLayer->getNextLayer()->getInputSize(), currentLayer->getNextLayer()->getOutputSize(), true);
        

            MatrixOps::multiply(currentLayer->getNextLayer()->getDelta(), wT, currentLayer->getDelta(), 
                               1, currentLayer->getNextLayer()->getOutputSize(), 
                               currentLayer->getNextLayer()->getOutputSize(), currentLayer->getNextLayer()->getInputSize(), true);
            applyAntiActivation(currentLayer);

            MatrixOps::elementWiseMultiply(currentLayer->getPreActivation(), currentLayer->getDelta(), currentLayer->getDelta(), 
                                            1, currentLayer->getNextLayer()->getInputSize(), true);
            
            cudaMalloc(&wT_current, currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float));
            cudaMemcpy(wT_current, currentLayer->getWeights(), currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float), cudaMemcpyDeviceToDevice);
            MatrixOps::transpose(wT_current, currentLayer->getWeights(), currentLayer->getInputSize(), currentLayer->getOutputSize(), true);
            
            cudaMalloc(&dlossW, currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float));
            cudaMemset(dlossW, 0, currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float));
            MatrixOps::multiply(currentLayer->getDelta(), wT_current, dlossW, 
                                currentLayer->getOutputSize(), 1, 
                                currentLayer->getInputSize(), currentLayer->getOutputSize(), true);
            cudaFree(wT_current);
            MatrixOps::add(currentLayer->getWeightGradients(), dlossW, currentLayer->getWeightGradients(), 
                           currentLayer->getInputSize(), currentLayer->getOutputSize(), true);
            MatrixOps::add(currentLayer->getBiasGradients(), currentLayer->getDelta(), currentLayer->getBiasGradients(), 
                          1, currentLayer->getOutputSize(), true);
            cudaFree(dlossW);
            cudaFree(wT);
            cudaFree(delta);
        }
        currentLayer = prevLayer;
    }
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

void NeuralNet::applyGradients(float learningRate, int batchSize) {
    Layer* currentLayer = outputLayer;
    
    while (currentLayer != inputLayer) {
        // Scale gradients by learning rate
        MatrixOps::scalerMultiplication(currentLayer->getWeightGradients(), 
                                      currentLayer->getWeightGradients(),
                                      1/batchSize,
                                      currentLayer->getInputSize(), 
                                      currentLayer->getOutputSize(), 
                                      true);
        
        MatrixOps::scalerMultiplication(currentLayer->getBiasGradients(), 
                                      currentLayer->getBiasGradients(),
                                      1/batchSize,
                                      1, 
                                      currentLayer->getOutputSize(), 
                                      true);

        // Debug: Print weight and bias gradients
        float* h_weightGrads = new float[currentLayer->getInputSize() * currentLayer->getOutputSize()];
        float* h_biasGrads = new float[currentLayer->getOutputSize()];
        
        cudaMemcpy(h_weightGrads, currentLayer->getWeightGradients(), 
                  currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float), 
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(h_biasGrads, currentLayer->getBiasGradients(),
                  currentLayer->getOutputSize() * sizeof(float),
                  cudaMemcpyDeviceToHost);

        std::cout << "Weight gradients (first 5): ";
        for (int i = 0; i < 5; i++) {
            std::cout << h_weightGrads[i] << " ";
        }
        std::cout << "\nBias gradients (first 5): ";
        for (int i = 0; i < 5; i++) {
            std::cout << h_biasGrads[i] << " ";
        }
        std::cout << std::endl;

        delete[] h_weightGrads;
        delete[] h_biasGrads;
        // Update weights with scaled gradients

        MatrixOps::add(currentLayer->getWeights(), 
                      currentLayer->getWeightGradients(), 
                      currentLayer->getWeights(),
                      currentLayer->getInputSize(), 
                      currentLayer->getOutputSize(), 
                      true);
        
        currentLayer = currentLayer->getPrevLayer();
    }
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