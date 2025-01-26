#ifndef LAYER_H
#define LAYER_H
#include "MatrixOps.h"

class Layer {
private:
    int inputSize;
    int outputSize;
    float* weights;
    float* biases;

    Layer* prevLayer;
    Layer* nextLayer;

public:
    Layer(int inputSize, int outputSize, Layer* prevLayer = nullptr);
    ~Layer();

    void initualizeWeights();
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
};

#endif