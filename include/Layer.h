#ifndef LAYER_H
#define LAYER_H
#include "MatrixOps.h"

class Layer {
private:
    // Core layer dimensions
    int inputSize;
    int outputSize;
    int batchSize = 0;

    // Primary network parameters
    float* weights;
    float* biases;
    float* output;

    // Layer connectivity
    Layer* prevLayer;
    Layer* nextLayer;

    // Gradient and backpropagation buffers
    float* weightGradients = nullptr;
    float* biasGradients = nullptr;
    float* delta = nullptr;
    float* preActivation = nullptr;
    float* weightsTransposed = nullptr;
    float* prevLayerOutputTransposed = nullptr;

    // Layer configuration
    std::string activationType;

public:
    // Constructor and destructor
    Layer(int inputSize, int outputSize, const std::string& activationType = "relu", Layer* prevLayer = nullptr);
    ~Layer();

    // Core layer operations
    void initializeWeights();
    void initialBiases();
    float* forward(const float* input, int batchSize);
    
    // Layer connectivity management
    void setNextLayer(Layer* next);
    void setPrevLayer(Layer* prev);
    Layer* getNextLayer() const;
    Layer* getPrevLayer() const;

    // Network parameter access
    float* getWeights() const;
    float* getBiases() const;
    int getInputSize() const;
    int getOutputSize() const;

    // Gradient and backpropagation management
    void setWeightGradients(float* gradients);
    float* getWeightGradients() const;
    float* getWeightsTransposed();
    void setBiasGradients(float* gradients);
    float* getBiasGradients() const;
    float* getDelta();
    float* getPreActivation();

    // Layer output handling
    const float* getPrevLayerOutput() const;
    float* getPrevLayerOutputTransposed();
    float* getOutput() const;
    void setOutput(float* out);

    // Configuration getters/setters
    int getBatchSize() const;
    void setBatchSize(int bSize);
    std::string getActivationType() const;
};

#endif