#include <iostream>
#include <cassert> // For test assertions
#include "LossOps.h"

void testMeanSquaredError() {
    const int size = 5;
    float yTrue[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float yPred[5] = {1.1f, 1.9f, 3.2f, 3.8f, 4.7f};

    float mse = LossOps::MeanSquaredError(yTrue, yPred, size);
    assert(mse - .034 < 1e-3);
    std::cout << "MSE test passed.\n";
}

void testCrossEntropyLoss() {
    const int batchSize = 2;
    const int numClasses = 3;
    float yTrue[6] = {1, 0, 0, 0, 1, 0};  // One-hot encoded labels
    float yPred[6] = {0.7, 0.2, 0.1, 0.1, 0.8, 0.1};  // Predicted probabilities

    float loss = LossOps::gpuCrossEntropyLoss(yTrue, yPred, batchSize, numClasses);
    assert(loss - .259 < 1e-4);
    std::cout << "Cross-Entropy Loss paassed." << std::endl;
}

int main() {
    testMeanSquaredError();
    testCrossEntropyLoss();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}