#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "Layer.h"
#include <vector>

class NeuralNet {
private:
    Layer* firstLayer;
    Layer* outputLayer;
    Layer* inputLayer;

    float* forward(const float* input, int batchSize); // Perform forward pass through all layers
    void backward(const float* predictions, const float* targets, int batchSize);
    float calculateLoss(const float* predictions, const float* targets, int batchSize);
    float* computeWeightGradient(Layer* currentLayer, const float* dLoss, int batchSize);
    float* propagateGradientBackward(Layer* currentLayer, const float* dLoss, int batchSize);
    float* computeBiasGradient(Layer* currentLayer, const float* dLoss, int batchSize);
    float* extractBatch(const float* fullData, const std::vector<int>& indices, 
                               int startIdx, int batchSize, int numFeatures);
    void applyGradients(float learningRate);

public:
    NeuralNet();
    ~NeuralNet();

    void NeuralNet::initialize(int samples, int features, int hiddenSize, int outputFeatures);   
    // void addLayer(int inputSize, int outputSize, const std::string& activation); // Add hidden layers
    void train(const float* inputs, const float* targets, int numSamples, int numFeatures, int batchSize, int epochs, float learningRate);

    Layer* getInputLayer() const;
    Layer* getOutputLayer() const;

};

#endif