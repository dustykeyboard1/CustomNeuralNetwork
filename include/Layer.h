#ifndef LAYER_H
#define LAYER_H
#include "MatrixOps.h"

class Layer {
private:
    int inputSize;
    int outputSize;
    float* weights;
    float* biases;
    float* output;

    Layer* prevLayer;
    Layer* nextLayer;

    float* weightGradients = nullptr;
    float* biasGradients = nullptr;
    float* weightsTransposed = nullptr;
    float* prevLayerOutputTransposed = nullptr;

    std::string activationType;

    int batchSize = 0;

public:
    Layer(int inputSize, int outputSize, const std::string& activationType = "relu", Layer* prevLayer = nullptr);
    ~Layer();

    void initializeWeights();
    void initialBiases();
    float* forward(const float* input, int batchSize);
    
    void setNextLayer(Layer* next);
    void setPrevLayer(Layer* prev);

    float* getWeights() const;
    float* getBiases() const;
    int getInputSize() const;
    int getOutputSize() const;
    Layer* getNextLayer() const;
    Layer* getPrevLayer() const;

    void setWeightGradients(float* gradients);
    float* getWeightGradients() const;
    float* getWeightsTransposed();

    const float* getPrevLayerOutput() const;
    float* getPrevLayerOutputTransposed();

    void setBiasGradients(float* gradients);
    float* getBiasGradients() const;

    float* getOutput() const;
    void setOutput(float* out);

    int getBatchSize() const;
    void setBatchSize(int bSize);

};

#endif