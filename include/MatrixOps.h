#ifndef MARTIX_OPS_H
#define MATRIX_OPS_H
#include <string>

namespace MatrixOps {
    // Matrix Addition
    void add(const float* A, const float* B, float* C, int rows, int cols); 
    void subtract(const float* A, const float* B, float* C, int rows, int cols);
    void multiply(const float* A, const float* B, float* C, int rows, int cols, int rowsB, int colsB);
    void divide(const float* A, const float* B, float* C, int rows, int cols);
    void transpose(const float* A, float* B, int rows, int cols);
    void scalerAddition(const float* A, float* B, const float k, int rows, int cols);
    void scalerMultiplication(const float* A, float* B, const float k, int rows, int cols);
    void Relu(const float* A, float* B, int rows, int cols);
    void Sigmoid(const float* A, float* B, int rows, int cols); 
    void Tanh(const float* A, float* B, int rows, int cols);
    void Softmax(const float* A, float* B, int rows, int cols); 
    void initializeWeights(float* weights, int rows, int cols, const std::string& initType = "uniform");
    void addBias(const float* output, const float* bias, float* result, int batchSize, int outputSize);
    void sumAcrossRows(const float* input, float* output, int rows, int cols);
    void reset();


}

#endif