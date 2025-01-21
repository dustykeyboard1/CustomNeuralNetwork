#ifndef MARTIX_OPS_H
#define MATRIX_OPS_H

namespace MatrixOps {
    // Matrix Addition
    void add(const float* A, const float* B, float* C, int rows, int cols); 
    void multiply(const float* A, const float* B, float* C, int rows, int cols, int rowsB, int colsB);
}

#endif