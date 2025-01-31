#ifndef MARTIX_OPS_H
#define MATRIX_OPS_H
#include <string>

namespace MatrixOps {
    void add(const float* A, const float* B, float* C, int rows, int cols, bool isGPU = false); 
    void subtract(const float* A, const float* B, float* C, int rows, int cols, bool isGPU = false);
    void multiply(const float* A, const float* B, float* C, int rows, int cols, int rowsB, int colsB, bool isGPU);
    void divide(const float* A, const float* B, float* C, int rows, int cols);
    void transpose(const float* A, float* B, int rows, int cols);
    void scalerAddition(const float* A, float* B, const float k, int rows, int cols);
    void scalerMultiplication(const float* A, float* B, float k, int rows, int cols, bool isGPU = false);
    void Relu(const float* A, float* B, int rows, int cols, bool isGPU = false);
    void Sigmoid(const float* A, float* B, int rows, int cols, bool isGPU = false);
    void Tanh(const float* A, float* B, int rows, int cols, bool isGPU = false);
    void Softmax(const float* A, float* B, int rows, int cols, bool isGPU = false);
    void initializeWeights(float* d_weights, int rows, int cols, const std::string& initType = "uniform");
    void addBias(const float* output, const float* bias, float* result, int batchSize, int outputSize, bool isGPU = false);
    void sumAcrossRows(const float* input, float* output, int rows, int cols, bool isGPU = false);
    void reset();
    void ReluGradient(const float* output, float* gradient, int rows, int cols, bool isGPU = false);
    void SigmoidGradient(const float* output, float* gradient, int rows, int cols, bool isGPU = false);
    void TanhGradient(const float* output, float* gradient, int rows, int cols, bool isGPU = false);
    void SoftmaxGradient(const float* output, float* gradient, int rows, int cols, bool isGPU = false);
    void clipValues(float* input, float min, float max, int rows, int cols, bool isGPU = false);
}

#endif