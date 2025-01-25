#include <iostream>
#include <cassert> // For test assertions
#include "MatrixOps.h"

void testAddition() {
    const int rows = 2, cols = 2;
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0};

    MatrixOps::add(A, B, C, rows, cols);

    float expected[4] = {6, 8, 10, 12};
    for (int i = 0; i < rows * cols; ++i) {
        assert(C[i] == expected[i]);
    }
    std::cout << "Addition passed.\n";
}

void testSubtraction() {
    const int rows = 2, cols = 2;
    float A[4] = {5, 6, 7, 8};
    float B[4] = {1, 2, 3, 4};
    float C[4] = {0};

    MatrixOps::subtract(A, B, C, rows, cols);

    float expected[4] = {4, 4, 4, 4};
    for (int i = 0; i < rows * cols; ++i) {
        assert(C[i] == expected[i]);
    }
    std::cout << "Subtraction passed.\n";
}

void testMultiplication() {
    const int rowsA = 2, colsA = 3, rowsB = 3, colsB = 2;
    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[6] = {7, 8, 9, 10, 11, 12};
    float C[4] = {0};

    MatrixOps::multiply(A, B, C, rowsA, colsA, rowsB, colsB);

    float expected[4] = {58, 64, 139, 154};
    for (int i = 0; i < rowsA * colsB; ++i) {
        assert(C[i] == expected[i]);
    }
    std::cout << "Multiplication passed.\n";
}

void testDivision() {
    const int rows = 2, cols = 2;
    float A[4] = {8, 12, 18, 24};
    float B[4] = {2, 3, 6, 4};
    float C[4] = {0};

    MatrixOps::divide(A, B, C, rows, cols);

    float expected[4] = {4, 4, 3, 6};
    for (int i = 0; i < rows * cols; ++i) {
        assert(C[i] == expected[i]);
    }
    std::cout << "Division passed.\n";
}

void testTranspose() {
    const int rows = 2, cols = 3;
    float A[6] = {1, 2, 3, 4, 5, 6};
    float B[6] = {0};

    MatrixOps::transpose(A, B, rows, cols);

    float expected[6] = {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < rows * cols; ++i) {
        assert(B[i] == expected[i]);
    }
    std::cout << "Transpose passed.\n";
}

void testScalerAddition() {
    const int rows = 2, cols = 2;
    float A[4] = {1, 2, 3, 4};
    float C[4] = {0};
    int k = 5;

    MatrixOps::scalerAddition(A, C, k, rows, cols);

    float expected[4] = {6, 7, 8, 9};
    for (int i = 0; i < rows * cols; ++i) {
        assert(C[i] == expected[i]);
    }
    std::cout << "Scaler Addition passed.\n";

    MatrixOps::scalerAddition(A, C, -k, rows, cols);

    float expectedB[4] = {-4, -3, -2, -1};
    for (int i = 0; i < rows * cols; ++i) {
        assert(C[i] == expectedB[i]);
    }
    std::cout << "Scaler Subtraction passed.\n";
}

void testScalerMultiplication() {
    const int rows = 2, cols = 2;
    float A[4] = {2, 4, 6, 8};
    float C[4] = {0};
    float k = 2;

    MatrixOps::scalerMultiplication(A, C, k, rows, cols);

    float expected[4] = {4, 8, 12, 16};
    for (int i = 0; i < rows * cols; ++i) {
        assert(C[i] == expected[i]);
    }
    std::cout << "Scaler Multiplication passed.\n";

    MatrixOps::scalerMultiplication(A, C, 1/k, rows, cols);
    float expectedB[4] = {1, 2, 3, 4};
    for (int i = 0; i < rows * cols; ++i) {
        assert(C[i] == expectedB[i]);
    }
    std::cout << "Scaler Reverse Multiplication passed.\n";
}

int main() {
    testAddition();
    testSubtraction();
    testMultiplication();
    testDivision();
    testTranspose();
    testScalerAddition();
    testScalerMultiplication();

    std::cout << "All tests passed successfully!\n";
    return 0;
}
