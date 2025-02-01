#ifndef MARTIX_OPS_H
#define MATRIX_OPS_H
#include <string>

namespace MatrixOps {
    // Basic matrix arithmetic operations
    void add(const float* A, const float* B, float* C, int rows, int cols, bool isGPU = false); 
    void subtract(const float* A, const float* B, float* C, int rows, int cols, bool isGPU = false);
    void multiply(const float* A, const float* B, float* C, int rows, int cols, int rowsB, int colsB, bool isGPU);
    void divide(const float* A, const float* B, float* C, int rows, int cols);
    
    // Matrix transformation operations
    void transpose(float* output, const float* input, int rows, int cols, bool isGPU);
    void scalerAddition(const float* A, float* B, const float k, int rows, int cols);
    void scalerMultiplication(const float* A, float* B, float k, int rows, int cols, bool isGPU = false);
    
    // Neural network activation functions
    void Relu(const float* A, float* B, int rows, int cols, bool isGPU = false);
    void Sigmoid(const float* A, float* B, int rows, int cols, bool isGPU = false);
    void Tanh(const float* A, float* B, int rows, int cols, bool isGPU = false);
    void Softmax(const float* A, float* B, int rows, int cols, bool isGPU = false);
    
    // Network initialization and bias operations
    void initializeWeights(float* d_weights, int rows, int cols, const std::string& initType = "uniform");
    void addBias(const float* output, const float* bias, float* result, int batchSize, int outputSize, bool isGPU = false);
    
    // Utility operations
    void sumAcrossRows(const float* input, float* output, int rows, int cols, bool isGPU = false);
    void reset();
    
    // Gradient computation for activation functions
    void ReluGradient(const float* output, float* gradient, int rows, int cols, bool isGPU = false);
    void SigmoidGradient(const float* output, float* gradient, int rows, int cols, bool isGPU = false);
    void TanhGradient(const float* output, float* gradient, int rows, int cols, bool isGPU = false);
    void SoftmaxGradient(const float* output, float* gradient, int rows, int cols, bool isGPU = false);
    
    // Advanced matrix operations
    void clipValues(float* input, float min, float max, int rows, int cols, bool isGPU = false);
    void elementWiseMultiply(const float* A, const float* B, float* C, int rows, int cols, bool isGPU = false);
    float computeL2Norm(const float* A, int rows, int cols, bool isGPU = false);
}

#endif