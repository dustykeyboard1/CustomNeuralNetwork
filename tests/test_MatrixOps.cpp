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

    MatrixOps::multiply(A, B, C, rowsA, colsA, rowsB, colsB, false);

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

void testRelu() {
    const int rows = 2, cols = 3;
    float A[6] = {-1, 2, -3, 4, -5, 6};
    float B[6] = {0};
    float expected[6] = {0, 2, 0, 4, 0, 6};

    MatrixOps::Relu(A, B, rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        assert(B[i] == expected[i]);
    }
    std::cout << "Relu Activation passed.\n";


}

void testSigmoid() {
    const int rows = 2, cols = 3;
    float A[6] = {-1, 2, -3, 4, -5, 6}; 
    float B[6] = {0};                  
    float expected[6] = {
        0.268941f, 0.880797f, 0.047426f,
        0.982014f, 0.006693f, 0.997527f
    }; // Expected Sigmoid values

    MatrixOps::Sigmoid(A, B, rows, cols);

    
    for (int i = 0; i < rows * cols; ++i) {
        assert(fabs(B[i] - expected[i]) < 1e-5); 
    }

    std::cout << "Sigmoid Activation passed.\n";
}

void testTanh() {
    const int rows = 2, cols = 3;
    float A[6] = {-1, 0, 1, -2, 2, 3}; 
    float B[6] = {0};                 
    float expected[6] = {
        -0.761594f, 0.000000f, 0.761594f,
        -0.964028f, 0.964028f, 0.995055f
    }; 

    MatrixOps::Tanh(A, B, rows, cols); 

    for (int i = 0; i < rows * cols; ++i) {
        assert(fabs(B[i] - expected[i]) < 1e-5); 
    }

    std::cout << "Tanh Activation passed.\n";
}

void testSoftmax() {
    const int rows = 1, cols = 3;
    float A[3] = {2, 1, 0.1f}; 
    float B[3] = {0};                 
    float expected[3] = {0.659f, 0.242f, 0.0986f}; 

    MatrixOps::Softmax(A, B, rows, cols); 

    for (int i = 0; i < rows * cols; ++i) {
        assert(fabs(B[i] - expected[i]) < 1e-3);  
    }

    std::cout << "Softmax Activation passed.\n";
}

void testInitialization() {
    const int rows = 5, cols = 10;
    float weights[rows * cols] = {0};


    MatrixOps::initializeWeights(weights, rows, cols, "xavier");
    std::cout << "Xaiver Init Matrix: \n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) { 
            std::cout << weights[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    std::fill(weights, weights + (rows * cols), 0.0f);
    MatrixOps::initializeWeights(weights, rows, cols, "he");
    std::cout << "He Init Matrix: \n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) { 
            std::cout << weights[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

    std::fill(weights, weights + (rows * cols), 0.0f);
    MatrixOps::initializeWeights(weights, rows, cols);
    std::cout << "Default Init Matrix: \n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) { 
            std::cout << weights[i * cols + j] << " ";
        }
        std::cout << "\n";
    }

}

void testAddBias() {
    const int batchSize = 2;
    const int outputSize = 3;

    float output[6] = {
        1.0f, 2.0f, 3.0f,  // Row 1
        4.0f, 5.0f, 6.0f   // Row 2
    };

    float bias[3] = {0.1f, 0.2f, 0.3f}; // Bias vector
    float result[6] = {0};              // Output after adding bias
    float expected[6] = {
        1.1f, 2.2f, 3.3f,  // Row 1 + Bias
        4.1f, 5.2f, 6.3f   // Row 2 + Bias
    };

    MatrixOps::addBias(output, bias, result, batchSize, outputSize);

    // Validate the result
    for (int i = 0; i < batchSize * outputSize; ++i) {
        assert(fabs(result[i] - expected[i]) < 1e-3);
        
    }
    std::cout << "Bias Addition passed.\n";
}


void testSumAcrossRows() {
    const int rows = 3, cols = 4;
    float input[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    float output[cols] = {0}; // Expected: {15, 18, 21, 24}

    MatrixOps::sumAcrossRows(input, output, rows, cols);

    // Check output
    float expected[4] = {15.0f, 18.0f, 21.0f, 24.0f};
    for (int i = 0; i < cols; ++i) {
        assert(fabs(output[i] - expected[i]) < 1e-3);
    }
    std::cout << "SumAcrossRows passed.\n";
}




int main() {
    MatrixOps::reset();
    testAddition();
    testSubtraction();
    testMultiplication();
    testDivision();
    testTranspose();
    testScalerAddition();
    testScalerMultiplication();
    testSigmoid();
    testRelu();
    testTanh();
    testSoftmax();
    // testInitialization();
    testAddBias();
    testSumAcrossRows();

    std::cout << "All tests passed successfully!\n";
    return 0;
}
