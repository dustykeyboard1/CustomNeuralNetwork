//------------------------------------------------------------------------------
// Neural Network Header - Defines core neural network functionality
//------------------------------------------------------------------------------
#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "Layer.h"
#include <vector>

class NeuralNet {
private:
    // Core network components
    Layer* firstLayer;
    Layer* outputLayer; 
    Layer* inputLayer;

    // Data preprocessing
    std::vector<float> featureMeans;
    std::vector<float> featureStds;

    // Forward propagation
    void forward();
    void applyActivation(Layer* currentLayer);
    void applyAntiActivation(Layer* currentLayer);

    // Backward propagation and gradient computation
    void backward(const float* predictions, const float* targets, int batchSize);
    float calculateLoss(const float* predictions, const float* targets, int batchSize);
    float* computeWeightGradient(Layer* currentLayer, const float* dLoss, int batchSize);
    float* propagateGradientBackward(Layer* currentLayer, const float* dLoss, int batchSize);
    float* computeBiasGradient(Layer* currentLayer, const float* dLoss, int batchSize);
    void clipGradients(Layer* currentLayer, float clipThreshold);
    void applyGradients(float learningRate, int batchSize);

    // Data handling and preprocessing
    void extractBatch(const float* fullData, const std::vector<int>& indices, 
                     int startIdx, int batchSize, int numFeatures);
    void standardizeData(const float* rawData, float* standardizedData, 
                        int numSamples, int numFeatures);
    void reverseStandardization(const float* standardizedData, float* originalScaleData,
                              int numSamples, int numFeatures);

    // Debug utilities
    void debugPrintGPUArray(const char* name, const float* d_array, int rows, int cols);

public:
    // Constructor/Destructor
    NeuralNet();
    ~NeuralNet();

    // Network setup and training
    void NeuralNet::initialize(int inputSize, int* Nuerons, int hiddenLayers, int outputFeatures);   
    void train(const float* trainingData,
               int numDays,
               int lookback,
               int numFeatures,
               int numPredictions,
               int batchSize,
               float learningRate, 
               int numEpochs,
               const int* targetIndices);

    // Network access methods
    void setInput(const float* input, int rows, int cols);
    Layer* getInputLayer() const;
    Layer* getOutputLayer() const;
};
#endif