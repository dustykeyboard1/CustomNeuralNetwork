#ifndef MARTIX_OPS_H
#define MATRIX_OPS_H

namespace MatrixOps {
    // Matrix Addition
    void add(const float* A, const float* B, float* C, int rows, int cols); 
    void subtract(const float* A, const float* B, float* C, int rows, int cols);
    void multiply(const float* A, const float* B, float* C, int rows, int cols, int rowsB, int colsB);
    void divide(const float* A, const float* B, float* C, int rows, int cols);
    void transpose(const float* A, float* B, int rows, int cols);
    void scalerAddition(const float* A, float* B, const float k, int rows, int cols);
    void scalerMultiplication(const float* A, float* B, const float k, int rows, int cols);
}

#endif