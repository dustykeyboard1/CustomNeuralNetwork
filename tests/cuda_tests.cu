#include <iostream>
#include <cassert>
#include "MatrixOps.h"

void testAddBias() {
    const int batchSize = 2;
    const int outputSize = 3;

    float hostOutput[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float hostBias[3] = {0.1f, 0.2f, 0.3f};
    float hostResult[6] = {0};
    float expected[6] = {1.1f, 2.2f, 3.3f, 4.1f, 5.2f, 6.3f};

    float *d_output, *d_bias, *d_result;
    cudaMalloc(&d_output, sizeof(hostOutput));
    cudaMalloc(&d_bias, sizeof(hostBias));
    cudaMalloc(&d_result, sizeof(hostResult));

    cudaMemcpy(d_output, hostOutput, sizeof(hostOutput), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, hostBias, sizeof(hostBias), cudaMemcpyHostToDevice);

    MatrixOps::addBias(d_output, d_bias, d_result, batchSize, outputSize);

    cudaMemcpy(hostResult, d_result, sizeof(hostResult), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batchSize * outputSize; ++i) {
        assert(fabs(hostResult[i] - expected[i]) < 1e-5);
    }

    std::cout << "addBias passed.\n";

    cudaFree(d_output);
    cudaFree(d_bias);
    cudaFree(d_result);
}

int main() {
    testAddBias();
    return 0;
}
