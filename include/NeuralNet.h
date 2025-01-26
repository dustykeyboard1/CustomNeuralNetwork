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
    void backward(const float* predictions, const float* targets, int batchSize, float learningRate);
    float computeLoss(const float* predictions, const float* targets, int batchSize, int outputSize);
    float* computeWeightGradient(Layer* currentLayer, const float* dLoss, int batchSize);
    float* propagateGradientBackward(Layer* currentLayer, const float* dLoss, int batchSize);
    float* computeBiasGradient(Layer* currentLayer, const float* dLoss, int batchSize);

public:
    NeuralNet();
    ~NeuralNet();

    void initialize(int inputSize, int inputFeatures, int outputSize, int outputFeatures);
    void addLayer(int inputSize, int outputSize, const std::string& activation); // Add hidden layers
    void train(const float* inputs, const float* targets, int batchSize, int numEpochs, float learningRate);

    Layer* getInputLayer() const;
    Layer* getOutputLayer() const;

};

#endif