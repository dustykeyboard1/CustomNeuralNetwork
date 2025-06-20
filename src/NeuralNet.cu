#include "LossOps.h"
#include "NeuralNet.h"
#include "Utils.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

// Constructor and destructor
NeuralNet::NeuralNet() : inputLayer(nullptr), outputLayer(nullptr), firstLayer(nullptr) {}

NeuralNet::~NeuralNet()
{
    Layer *current = inputLayer;
    while (current != nullptr)
    {
        Layer *next = current->getNextLayer();
        delete current;
        current = next;
    }
}

// Network initialization
void NeuralNet::initialize(int inputSize, const std::vector<int> &neurons, int outputFeatures)
{
    inputLayer = new Layer(1, inputSize, "relu");
    Layer *prevLayer = inputLayer;
    Layer *currentLayer;

    for (int i = 0; i < neurons.size(); ++i)
    {
        currentLayer = new Layer(prevLayer->getOutputSize(), neurons[i], "relu");
        prevLayer->setNextLayer(currentLayer);
        currentLayer->setPrevLayer(prevLayer);
        prevLayer = currentLayer;
    }

    outputLayer = new Layer(prevLayer->getOutputSize(), outputFeatures, "softmax");
    outputLayer->setPrevLayer(prevLayer);
    prevLayer->setNextLayer(outputLayer);
}

// Main training loop
std::vector<float> NeuralNet::train(const float *inputs,   // flattened input data
                                    const float *targets,  // flattened target data
                                    int numSamples, int inputSize, int outputSize, int batchSize,
                                    float learningRate, int numEpochs, float clipThreshold,
                                    float decayRate, const std::vector<int> &batchIndices)
{
    float loss;
    float epochLoss;
    std::vector<float> epochLosses;

    int numBatchesPerEpoch = numSamples / batchSize;
    float currentLearningRate = learningRate;

    for (int epoch = 0; epoch < numEpochs; ++epoch)
    {
        epochLoss = 0.0f;
        for (int batch = 0; batch < numBatchesPerEpoch; ++batch)
        {
            // Reset gradients for each batch
            Layer *layer = inputLayer->getNextLayer();
            while (layer != nullptr)
            {
                cudaMemset(layer->getWeightGradients(), 0,
                           layer->getInputSize() * layer->getOutputSize() * sizeof(float));
                cudaMemset(layer->getBiasGradients(), 0, layer->getOutputSize() * sizeof(float));
                layer = layer->getNextLayer();
            }

            // Process each sample in batch
            loss = 0.0f;
            for (int i = 0; i < batchSize; i++)
            {
                int currentT = batchIndices[batch * batchSize + i];

                // Get sequence [T-lookback+1, ..., T]
                const float *inputSample = &inputs[currentT * inputSize];
                cudaMemcpy(inputLayer->getOutput(), inputSample, inputSize * sizeof(float),
                           cudaMemcpyHostToDevice);

                forward();

                // Get targets from T+1, using targetIndices to select specific features
                const float *targetSample = &targets[currentT * outputSize];

                loss += calculateLoss(outputLayer->getOutput(), targetSample, 1);
                backward(outputLayer->getOutput(), targetSample, 1);
            }
            applyGradients(currentLearningRate, batchSize, clipThreshold);
            epochLoss += loss;
        }

        currentLearningRate = learningRate * expf(-decayRate * float(epoch));

        float avgEpochLoss = epochLoss / (float(numBatchesPerEpoch) * float(batchSize));
        epochLosses.push_back(avgEpochLoss);
        Utils::printProgress(epoch + 1, numEpochs, avgEpochLoss);
    }

    return epochLosses;
}

// Forward pass through network
void NeuralNet::forward()
{
    Layer *currentLayer = inputLayer->getNextLayer();
    int layerIndex = 0;
    Layer *prevLayer = nullptr;

    while (currentLayer != nullptr)
    {
        prevLayer = currentLayer->getPrevLayer();
        cudaMemset(currentLayer->getOutput(), 0, 1 * currentLayer->getOutputSize() * sizeof(float));

        // Matrix multiplication and bias addition
        MatrixOps::multiply(prevLayer->getOutput(), currentLayer->getWeights(),
                            currentLayer->getOutput(), 1, prevLayer->getOutputSize(),
                            currentLayer->getInputSize(), currentLayer->getOutputSize(), true);

        MatrixOps::addBias(currentLayer->getOutput(), currentLayer->getBiases(),
                           currentLayer->getOutput(), 1, currentLayer->getOutputSize(), true);

        // Apply activation function except for output layer
        if (currentLayer->getNextLayer() != nullptr)
        {
            cudaMemcpy(currentLayer->getPreActivation(), currentLayer->getOutput(),
                       currentLayer->getOutputSize() * sizeof(float), cudaMemcpyDeviceToDevice);
            applyActivation(currentLayer);
        }

        currentLayer = currentLayer->getNextLayer();
        layerIndex++;
    }
}

// Activation functions
void NeuralNet::applyActivation(Layer *currentLayer)
{
    if (currentLayer->getActivationType() == "relu")
    {
        MatrixOps::Relu(currentLayer->getOutput(), currentLayer->getOutput(), 1,
                        currentLayer->getOutputSize(), true);
    }
    else if (currentLayer->getActivationType() == "softmax")
    {
        MatrixOps::Softmax(currentLayer->getOutput(), currentLayer->getOutput(), 1,
                           currentLayer->getOutputSize(), true);
    }
    else if (currentLayer->getActivationType() == "sigmoid")
    {
        MatrixOps::Sigmoid(currentLayer->getOutput(), currentLayer->getOutput(), 1,
                           currentLayer->getOutputSize(), true);
    }
    else if (currentLayer->getActivationType() == "tanh")
    {
        MatrixOps::Tanh(currentLayer->getOutput(), currentLayer->getOutput(), 1,
                        currentLayer->getOutputSize(), true);
    }
}

void NeuralNet::applyAntiActivation(Layer *currentLayer)
{
    if (currentLayer->getActivationType() == "relu")
    {
        MatrixOps::ReluGradient(currentLayer->getPreActivation(), currentLayer->getPreActivation(),
                                1, currentLayer->getOutputSize(), true);
    }
    else if (currentLayer->getActivationType() == "sigmoid")
    {
        MatrixOps::SigmoidGradient(currentLayer->getPreActivation(),
                                   currentLayer->getPreActivation(), 1,
                                   currentLayer->getOutputSize(), true);
    }
    else if (currentLayer->getActivationType() == "tanh")
    {
        MatrixOps::TanhGradient(currentLayer->getPreActivation(), currentLayer->getPreActivation(),
                                1, currentLayer->getOutputSize(), true);
    }
    else if (currentLayer->getActivationType() == "softmax")
    {
        return;
    }
}

// Loss calculation
float NeuralNet::calculateLoss(const float *predictions, const float *targets, int batchSize)
{
    int outputSize = outputLayer->getOutputSize();
    int size = batchSize * outputSize;
    return LossOps::MeanSquaredError(targets, predictions, size, true);
}

// Backward pass through network
void NeuralNet::backward(const float *predictions, const float *targets, int batchSize)
{
    Layer *currentLayer = outputLayer;
    float *dlossW = nullptr;
    float *wT;
    float *aT;

    while (currentLayer != inputLayer)
    {
        Layer *prevLayer = currentLayer->getPrevLayer();

        // Output layer gradients
        if (currentLayer == outputLayer)
        {
            LossOps::computeError(targets, predictions, currentLayer->getDelta(),
                                  currentLayer->getOutputSize(), true);

            cudaMalloc(&aT, prevLayer->getOutputSize() * sizeof(float));
            cudaMemcpy(aT, prevLayer->getOutput(), prevLayer->getOutputSize() * sizeof(float),
                       cudaMemcpyDeviceToDevice);

            MatrixOps::transpose(aT, aT, 1, prevLayer->getOutputSize(), true);

            cudaMalloc(&dlossW, currentLayer->getInputSize() * currentLayer->getOutputSize() *
                                    sizeof(float));
            cudaMemset(
                dlossW, 0,
                currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float));

            MatrixOps::multiply(aT, currentLayer->getDelta(), dlossW, prevLayer->getOutputSize(), 1,
                                1, currentLayer->getOutputSize(), true);
            MatrixOps::add(currentLayer->getWeightGradients(), dlossW,
                           currentLayer->getWeightGradients(), prevLayer->getInputSize(),
                           currentLayer->getOutputSize(), true);

            MatrixOps::add(currentLayer->getBiasGradients(), currentLayer->getDelta(),
                           currentLayer->getBiasGradients(), 1, currentLayer->getOutputSize(),
                           true);

            cudaFree(dlossW);
            cudaFree(aT);

            // Hidden layer gradients
        }
        else
        {
            Layer *nextLayer = currentLayer->getNextLayer();
            cudaMalloc(&wT, nextLayer->getInputSize() * nextLayer->getOutputSize() * sizeof(float));
            cudaMemcpy(wT, nextLayer->getWeights(),
                       nextLayer->getInputSize() * nextLayer->getOutputSize() * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            MatrixOps::transpose(wT, wT, nextLayer->getInputSize(), nextLayer->getOutputSize(),
                                 true);

            MatrixOps::multiply(nextLayer->getDelta(), wT, currentLayer->getDelta(), 1,
                                nextLayer->getOutputSize(), nextLayer->getOutputSize(),
                                nextLayer->getInputSize(), true);

            applyAntiActivation(currentLayer);

            MatrixOps::elementWiseMultiply(currentLayer->getPreActivation(),
                                           currentLayer->getDelta(), currentLayer->getDelta(), 1,
                                           nextLayer->getInputSize(), true);

            cudaMalloc(&aT, prevLayer->getOutputSize() * sizeof(float));
            cudaMemcpy(aT, prevLayer->getOutput(), prevLayer->getOutputSize() * sizeof(float),
                       cudaMemcpyDeviceToDevice);
            MatrixOps::transpose(aT, aT, 1, prevLayer->getOutputSize(), true);

            cudaMalloc(&dlossW, currentLayer->getInputSize() * currentLayer->getOutputSize() *
                                    sizeof(float));
            cudaMemset(
                dlossW, 0,
                currentLayer->getInputSize() * currentLayer->getOutputSize() * sizeof(float));
            MatrixOps::multiply(aT, currentLayer->getDelta(), dlossW, prevLayer->getOutputSize(), 1,
                                1, currentLayer->getOutputSize(), true);

            MatrixOps::add(currentLayer->getWeightGradients(), dlossW,
                           currentLayer->getWeightGradients(), currentLayer->getInputSize(),
                           currentLayer->getOutputSize(), true);
            MatrixOps::add(currentLayer->getBiasGradients(), currentLayer->getDelta(),
                           currentLayer->getBiasGradients(), 1, currentLayer->getOutputSize(),
                           true);
            cudaFree(dlossW);
            cudaFree(aT);
        }
        currentLayer = prevLayer;
    }
}

// Update weights and biases with gradients
void NeuralNet::applyGradients(float learningRate, int batchSize, float clipThreshold)
{
    Layer *currentLayer = outputLayer;

    while (currentLayer != inputLayer)
    {
        MatrixOps::scalerMultiplication(
            currentLayer->getWeightGradients(), currentLayer->getWeightGradients(),
            (1.0f / batchSize) * learningRate, currentLayer->getInputSize(),
            currentLayer->getOutputSize(), true);

        MatrixOps::scalerMultiplication(
            currentLayer->getBiasGradients(), currentLayer->getBiasGradients(),
            (1.0f / batchSize) * learningRate, 1, currentLayer->getOutputSize(), true);

        clipGradients(currentLayer, clipThreshold);

        MatrixOps::subtract(currentLayer->getWeights(), currentLayer->getWeightGradients(),
                            currentLayer->getWeights(), currentLayer->getInputSize(),
                            currentLayer->getOutputSize(), true);

        MatrixOps::subtract(currentLayer->getBiases(), currentLayer->getBiasGradients(),
                            currentLayer->getBiases(), 1, currentLayer->getOutputSize(), true);

        currentLayer = currentLayer->getPrevLayer();
    }
}

// Gradient clipping to prevent explosion
void NeuralNet::clipGradients(Layer *currentLayer, float clipThreshold)
{
    float L2Norm =
        MatrixOps::computeL2Norm(currentLayer->getWeightGradients(), currentLayer->getInputSize(),
                                 currentLayer->getOutputSize(), true);
    if (L2Norm > clipThreshold)
    {
        float scale = clipThreshold / L2Norm;
        MatrixOps::scalerMultiplication(
            currentLayer->getWeightGradients(), currentLayer->getWeightGradients(), scale,
            currentLayer->getInputSize(), currentLayer->getOutputSize(), true);
    }

    L2Norm = MatrixOps::computeL2Norm(currentLayer->getBiasGradients(), 1,
                                      currentLayer->getOutputSize(), true);
    if (L2Norm > clipThreshold)
    {
        float scale = clipThreshold / L2Norm;
        MatrixOps::scalerMultiplication(currentLayer->getBiasGradients(),
                                        currentLayer->getBiasGradients(), scale, 1,
                                        currentLayer->getOutputSize(), true);
    }
}

// Getter methods
Layer *NeuralNet::getInputLayer() const
{
    return inputLayer;
}

Layer *NeuralNet::getOutputLayer() const
{
    return outputLayer;
}

// Debug utilities
void NeuralNet::debugPrintGPUArray(const char *name, const float *d_array, int rows, int cols)
{
    int totalElements = rows * cols;
    float *h_array = new float[totalElements];
    cudaMemcpy(h_array, d_array, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "[DEBUG] " << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << h_array[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "-------------------\n";

    delete[] h_array;
}

void NeuralNet::setInput(const float *input, int rows, int cols)
{
    int flattenedSize = rows * cols;

    if (flattenedSize != inputLayer->getOutputSize())
    {
        std::cerr << "Error: Input size mismatch. Expected " << inputLayer->getOutputSize()
                  << " but got " << flattenedSize << " (rows: " << rows << ", cols: " << cols << ")"
                  << std::endl;
        return;
    }

    float *d_flattenedInput;
    cudaMalloc(&d_flattenedInput, flattenedSize * sizeof(float));

    for (int i = 0; i < rows; ++i)
    {
        cudaMemcpy(d_flattenedInput + (i * cols), input + (i * cols), cols * sizeof(float),
                   cudaMemcpyHostToDevice);
    }

    cudaMemcpy(inputLayer->getOutput(), d_flattenedInput, flattenedSize * sizeof(float),
               cudaMemcpyDeviceToDevice);

    cudaFree(d_flattenedInput);

    std::cout << "Set input layer with flattened data (" << rows << "x" << cols << " -> 1x"
              << flattenedSize << "):" << std::endl;
    debugPrintGPUArray("Flattened Input", inputLayer->getOutput(), 1, flattenedSize);
}

float NeuralNet::validate(const float *batchedInputs, const float *batchedTargets, int numSamples,
                          int numFeatures, int numPredictions, int lookback)
{
    float totalLoss = 0.0f;

    for (int i = 0; i < numSamples; ++i)
    {
        // Copy one input sequence to GPU
        cudaMemcpy(inputLayer->getOutput(), batchedInputs + i * lookback * numFeatures,
                   lookback * numFeatures * sizeof(float), cudaMemcpyHostToDevice);

        forward();

        // Grab target slice
        const float *targets = batchedTargets + i * numPredictions;

        totalLoss += calculateLoss(outputLayer->getOutput(), targets, 1);
    }

    return totalLoss / static_cast<float>(numSamples);
}
