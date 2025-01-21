#include "MatrixOps.h"
#include <iostream>

int main() {
    const int rowsA = 2, colsA = 3; // A: 2x3
    const int rowsB = 3, colsB = 4; // B: 3x4
    const int rowsC = rowsA, colsC = colsB; // C: 2x4

    float A[6] = {
        1, 2, 3,
        4, 5, 6
    };

    float B[12] = {
        7,  8,  9, 10,
        11, 12, 13, 14,
        15, 16, 17, 18
    };

    float C[8] = {0};

    // MatrixOps::add(A,B,C, rows, cols);
    MatrixOps::multiply(A,B,C, rowsA, colsA, rowsB, colsB);

    std::cout << "Matrix Multiplication Result:\n";
    for (int i = 0; i < rowsA; ++i) {
        for(int j = 0; j < colsB; ++j) {
            std::cout << C[i * colsB + j] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}