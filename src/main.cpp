#include "MatrixOps.h"
#include <iostream>

int main() {
    const int rows = 2, cols = 2;
    float A[4] = {1,2,3,4};
    float B[4] = {5,6,7,8}; 
    float C[4] = {0};

    MatrixOps::add(A,B,C, rows, cols);

    std::cout << "Matrix Addition Result:\n";
    for (int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            std::cout << C[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    return 0;
}