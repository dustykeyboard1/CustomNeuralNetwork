#include "Layer.h"
#include <iostream>

Layer::Layer(int inputSize, int outputSize, const std::string& activationType, Layer* prevLayer)
    : inputSize(inputSize), outputSize(outputSize), activationType(activationType), prevLayer(prevLayer), nextLayer(nullptr) {
        size_t weightSize = inputSize * outputSize * sizeof(float);
        size_t biasSize = outputSize * sizeof(float);

        cudaMalloc(&weights, weightSize);
        cudaMalloc(&biases, biasSize);

        size_t outputSizeBytes = inputSize * outputSize * sizeof(float);
        if (prevLayer != nullptr) {
            outputSizeBytes = prevLayer->getInputSize() * outputSize * sizeof(float);
        } 

        cudaMalloc(&output, outputSizeBytes);

        initializeWeights();
        initialBiases();

        std::cout << "Initialized layer with " << inputSize << " inputs and " << outputSize << " outputs.\n";
}
         
Layer::~Layer() {
    cudaFree(weights);
    cudaFree(biases);
    if (weightGradients) cudaFree(weightGradients);
    if (biasGradients) cudaFree(biasGradients);
    if (weightsTransposed) cudaFree(weightsTransposed);
    if (prevLayerOutputTransposed) cudaFree(prevLayerOutputTransposed);
}


void Layer::initializeWeights() {
    MatrixOps::initializeWeights(weights, inputSize, outputSize, "xavier");
    // std::cout << "[DEBUG] Layer Init Weights (first 5 values): ";
    // float hostWeights[5];
    // cudaMemcpy(hostWeights, weights, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 5; ++i) {
    //     std::cout << hostWeights[i] << " ";
    // }
    // std::cout << "\n";
}

void Layer::initialBiases() {
    MatrixOps::initializeWeights(biases, 1, outputSize, "uniform");
}

float* Layer::forward(const float* input, int batchSize) {
    // if (!input) {
    //     std::cerr << "[ERROR] Input pointer is null.\n";
    //     return nullptr; // Handle null input appropriately
    // }

    // this->batchSize = batchSize;

    // // Debug: Print the first few values of the input
    // std::cout << "[DEBUG] Verifying input to Matrix A in multiply (first 10 values): ";
    // float hostInputCheck[10];
    // cudaMemcpy(hostInputCheck, input, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << hostInputCheck[i] << " ";
    // }
    // std::cout << "\n";

    // // Allocate memory for the output matrix
    // size_t outputSizeBytes = batchSize * outputSize * sizeof(float);
    // cudaError_t err = cudaMalloc(&output, outputSizeBytes);
    // if (err != cudaSuccess) {
    //     std::cerr << "[ERROR] cudaMalloc for output failed: " << cudaGetErrorString(err) << "\n";
    //     return nullptr;
    // }

    // // Allocate memory for the weighted sum
    // float* weightedSum;
    // err = cudaMalloc(&weightedSum, outputSizeBytes);
    // if (err != cudaSuccess) {
    //     std::cerr << "[ERROR] cudaMalloc for weightedSum failed: " << cudaGetErrorString(err) << "\n";
    //     return nullptr;
    // }

    // // Debug: Matrix multiplication
    // std::cout << "[DEBUG] Performing matrix multiplication (input * weights)...\n";
    // MatrixOps::multiply(input, weights, weightedSum, batchSize, inputSize, inputSize, outputSize, true);

    // // Debug: Weighted sum before adding biases
    // std::cout << "[DEBUG] Weighted sum (first 5 values): ";
    // float hostWeightedSum[5];
    // cudaMemcpy(hostWeightedSum, weightedSum, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < 5; ++i) {
    //     std::cout << hostWeightedSum[i] << " ";
    // }
    // std::cout << "\n";

    // // Add biases to the weighted sum
    // std::cout << "[DEBUG] Adding biases to the weighted sum...\n";
    // MatrixOps::addBias(weightedSum, biases, output, batchSize, outputSize);

    // // Debug: Output before activation
    // std::cout << "[DEBUG] Output before activation (first 5 values): ";
    // float hostOutputBeforeActivation[5];
    // cudaMemcpy(hostOutputBeforeActivation, output, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < 5; ++i) {
    //     std::cout << hostOutputBeforeActivation[i] << " ";
    // }
    // std::cout << "\n";

    // Apply activation function
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

    // Debug: Output after activation
    // std::cout << "[DEBUG] Output after activation (first 5 values): ";
    // float hostOutput[5];
    // cudaMemcpy(hostOutput, output, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < 5; ++i) {
    //     std::cout << hostOutput[i] << " ";
    // }
    // std::cout << "\n";

    // Set the output for the layer
    // setOutput(output);
    return output;
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

void Layer::setWeightGradients(float* gradients) {
    if (weightGradients != nullptr) {
        cudaFree(weightGradients); // Free previously allocated memory, if any
    }
    weightGradients = gradients; // Store the new gradients
}

void Layer::setBiasGradients(float* gradients) {
    if (biasGradients != nullptr) {
        cudaFree(biasGradients); // Free previously allocated memory, if any
    }
    biasGradients = gradients; // Store the new gradients
}

float* Layer::getWeightsTransposed() {
    if (weightsTransposed == nullptr) {
        // Allocate memory for transposed weights
        size_t transposedSize = inputSize * outputSize * sizeof(float);
        cudaMalloc(&weightsTransposed, transposedSize);

        // Transpose weights
        MatrixOps::transpose(weights, weightsTransposed, outputSize, inputSize);
    }
    return weightsTransposed;
}

const float* Layer::getPrevLayerOutput() const {
    if (prevLayer == nullptr) {
        throw std::runtime_error("Previous layer is null. Cannot fetch output.");
    }
    return prevLayer->getOutput(); // Assumes `getOutput` returns the output of the layer
}

float* Layer::getPrevLayerOutputTransposed() {
    if (prevLayerOutputTransposed == nullptr) {
        // Allocate memory for transposed output
        size_t transposedSize = batchSize * inputSize * sizeof(float);
        cudaMalloc(&prevLayerOutputTransposed, transposedSize);

        // Transpose the previous layer's output
        MatrixOps::transpose(getPrevLayerOutput(), prevLayerOutputTransposed, batchSize, inputSize);
    }
    return prevLayerOutputTransposed;
}


float* Layer::getOutput() const {
    if (output == nullptr) {
        throw std::runtime_error("Layer output is not yet computed.");
    }
    return output;
}

void Layer::setOutput(float* out) {
    if (output != nullptr) {
        cudaFree(output); // Free previously allocated memory, if any
    }
    output = out; // Set the new output
}


int Layer::getBatchSize() const { 
    return batchSize; 
}
void Layer::setBatchSize(int bSize) { 
    batchSize = bSize; 
}

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

float* Layer::getWeights() const {
    return weights;
}

float* Layer::getBiases() const {
    return biases;
}

