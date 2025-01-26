#include "MatrixOps.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

namespace MatrixOps {
__global__ void addKernel(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    if (row < rows && col < cols) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void subtractKernel(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    if (row < rows && col < cols) {
        C[idx] = A[idx] - B[idx];
    }
}

__global__ void MultiplicationKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int rowsB, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rowsA && col < colsB) {
        float sum = 0.0f;

        for(int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

__global__ void DivideKernel(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int idx = row * cols + col; 

    if (row < rows && col < cols) {
        if (B[idx] != 0) {  
            C[idx] = A[idx] / B[idx];
        } else {
            C[idx] = 0; 
        }
    }
}

__global__ void TransposeKernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col; 
        int tranposed_idx = col * rows + row;
        B[tranposed_idx] = A[idx];
    }

} 

__global__ void ScalerAddKernel(const float* A, float* B, float k, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    if (row < rows && col < cols) {
        B[idx] = A[idx] + k;
    }
}

__global__ void ScalerMultiplyKernel(const float* A, float* B, float k, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    if (row < rows && col < cols) {
        B[idx] = A[idx] * k;
    }
}

__global__ void ReluKernel(const float* A, float* B, int rows, int cols) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    if (row < rows && col < cols) {
        B[idx] = max(A[idx], 0.0f);
    }
}

__global__ void SigmoidKernel(const float* A, float* B, int rows, int cols) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;
    if (row < rows && col < cols) {
        B[idx] = 1 / (1 + expf(-A[idx]));
    }
}

__global__ void TanhKernel(const float* A, float* B, int rows, int cols) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;

    if (row < rows && col < cols) {
        float exp_pos = expf(A[idx]);  
        float exp_neg = 1.0f / exp_pos; 
        B[idx] = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    }
}

__global__ void SoftMaxKernel(const float* A, float* B, int rows, int cols) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows) {
        float rowMax = -FLT_MAX;
        for (int col = 0; col < cols; ++col) {
            rowMax = max(rowMax, A[row * cols + col]);
        }

        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            float expVal = expf(A[row*cols+col]);
            B[row * cols + col] = expVal;
            sum += expVal;

        }
        for (int col = 0; col < cols; ++col) {
            B[row * cols + col] /= sum;
        }
    }
}

__global__ void initializeWeightsKernel(float* weights, int rows, int cols, curandState* states, int initTypeCode) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * cols + col;

    if (row < rows && col < cols) {

        curandState localState = states[idx];
        float n_in = rows;  
        float n_out = cols; 

        if (initTypeCode == 1) {
            float limit = sqrtf(6.0f / (n_in + n_out));
            weights[idx] = curand_uniform(&localState) * 2.0f * limit - limit;
        } else if (initTypeCode == 2) {
            float stddev = sqrtf(2.0f / n_in);
            weights[idx] = curand_normal(&localState) * stddev; 
        } else {
            // Default: Uniform small random values
            weights[idx] = curand_uniform(&localState) * 0.01f - 0.005f;
        }
        states[idx] = localState;
    }
}

__global__ void SumAcrossRowsKernel(const float* input, float* output, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int row = 0; row < rows; ++row) {
            sum += input[row * cols + col]; // Accumulate values from each row
        }
        output[col] = sum;

    }
}



__global__ void initializeCurandStates(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void AddBiasKernel(const float* output, const float* bias, float* result, int batchSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < batchSize * outputSize) {
        int col = idx % outputSize; // Column index
        result[idx] = output[idx] + bias[col];
    }
}

void add(const float* A, const float* B, float* C, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); 

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void subtract(const float* A, const float* B, float* C, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    subtractKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); 

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


}

void multiply(const float* A, const float* B, float* C, int rowsA, int colsA, int rowsB, int colsB) {
    if (colsA != rowsB) {
        std::cerr << "[ERROR] Matrix dimensions do not align for multiplication.\n";
        return;
    }
    if (A == nullptr) {
        std::cerr << "[ERROR] Host pointer A is null.\n";
        return;
    }

    size_t sizeA = rowsA * colsA * sizeof(float);
    std::cout << "[DEBUG] Calculated sizeA: " << sizeA << " bytes\n";

    size_t sizeB = rowsB * colsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);



    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaError_t allocErr = cudaMalloc(&d_A, sizeA);
    if (allocErr != cudaSuccess) {
        std::cerr << "[ERROR] cudaMalloc failed for d_A: " << cudaGetErrorString(allocErr) << "\n";
        return;
    }
    std::cout << "[DEBUG] Verifying device-side Matrix A after cudaMemcpy (first 10 values): ";
    float* hostAAfterMemcpy = new float[rowsA * colsA];
    cudaMemcpy(hostAAfterMemcpy, d_A, sizeA, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < 10 && i < rowsA * colsA; ++i) {
        std::cout << hostAAfterMemcpy[i] << " ";
    }
    std::cout << "\n";
    delete[] hostAAfterMemcpy;


    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);


    // Debug: Inspect A
    // float* hostA = new float[rowsA * colsA];
    // cudaMemcpy(hostA, d_A, sizeA, cudaMemcpyDeviceToHost);
    // std::cout << "[DEBUG] MAtrixOps: Matrix A (first 10 values): ";
    // for (int i = 0; i < 10 && i < rowsA * colsA; ++i) {
    //     std::cout << hostA[i] << " ";
    // }
    // std::cout << "\n";

    // Debug: Inspect B
    float* hostB = new float[rowsB * colsB];
    cudaMemcpy(hostB, d_B, sizeB, cudaMemcpyDeviceToHost);
    std::cout << "[DEBUG] MAtrixOps: Matrix B (first 10 values): ";
    for (int i = 0; i < 10 && i < rowsB * colsB; ++i) {
        std::cout << hostB[i] << " ";
    }
    std::cout << "\n";

    // Clean up debug arrays
    delete[] hostA;
    delete[] hostB;

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + 15) / 16, (rowsA + 15) / 16);

    MultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, rowsB,
                                                                colsB);

    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost); 


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void divide(const float* A, const float* B, float* C, int rows, int cols) {
    size_t sizeA = cols * rows * sizeof(float);
    size_t sizeB = cols * rows * sizeof(float);
    size_t sizeC = rows * cols * sizeof(float);



    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    DivideKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost); 

    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

void transpose(const float* A, float* B, int rows, int cols) {
    size_t sizeA = rows * cols * sizeof(float);
    size_t sizeB = cols * rows * sizeof(float);

    float *d_A, *d_B;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    TransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, rows, cols); 

    cudaMemcpy(B, d_B, sizeB, cudaMemcpyDeviceToHost); 
    cudaFree(d_A);
    cudaFree(d_B);
}

void scalerAddition(const float* A, float* B, const float k, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);


    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    ScalerAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, k, rows, cols);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost); 
    cudaFree(d_A);
    cudaFree(d_B);

}

void scalerMultiplication(const float* A, float* B, const float k, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);

    float *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);


    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    ScalerMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, k, rows, cols);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost); 

    cudaFree(d_A);
    cudaFree(d_B);

}
void Relu(const float* A, float* B, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    float *d_A, *d_B;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    ReluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, rows, cols);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost); 

    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);

}

void Sigmoid(const float* A, float* B, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    float *d_A, *d_B;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    SigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, rows, cols);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost); 

    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
}

void Tanh(const float* A, float* B, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    float *d_A, *d_B;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    TanhKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, rows, cols);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost); 

    cudaFree(d_A);
    cudaFree(d_B);
} 

void Softmax(const float* A, float* B, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);
    float *d_A, *d_B;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    SoftMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, rows, cols);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost); 

    cudaFree(d_A);
    cudaFree(d_B);
}

void initializeWeights(float* weights, int rows, int cols, const std::string& initType) {
    size_t size = rows * cols * sizeof(float);
    float *d_weights;

    cudaMalloc(&d_weights, size);

    cudaMemset(d_weights, 0, size);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    int initTypeCode = 0; 
    if (initType == "xavier") {
        initTypeCode = 1;
    } else if (initType == "he") {
        initTypeCode = 2;
    }

    int totalSize = rows * cols;
    curandState* d_states;
    cudaMalloc(&d_states, totalSize * sizeof(curandState));

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    initializeCurandStates<<<gridSize, blockSize>>>(d_states, time(0));

    initializeWeightsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights, rows, cols, d_states, initTypeCode);
    cudaDeviceSynchronize();

    cudaMemcpy(weights, d_weights, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_weights);
    cudaFree(d_states);
}


void MatrixOps::addBias(const float* output, const float* bias, float* result, int batchSize, int outputSize) {
    size_t totalElements = batchSize * outputSize * sizeof(float);
    size_t biasSize = outputSize * sizeof(float);

    float *d_output = nullptr, *d_bias = nullptr, *d_result = nullptr;

    // Allocate GPU memory
    cudaMalloc(&d_output, totalElements);
    cudaMalloc(&d_bias, biasSize);
    cudaMalloc(&d_result, totalElements);

    cudaMemcpy(d_output, output, totalElements, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, biasSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, totalElements, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize * outputSize + threadsPerBlock - 1) / threadsPerBlock;

    AddBiasKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_bias, d_result, batchSize, outputSize);
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, totalElements, cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(d_bias);
    cudaFree(d_result);
}


void MatrixOps::sumAcrossRows(const float* input, float* output, int rows, int cols) {
    float *d_input = nullptr, *d_output = nullptr;
    size_t inputSize = rows * cols * sizeof(float);
    size_t outputSize = cols * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, outputSize, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;

    SumAcrossRowsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
}


void MatrixOps::reset() {
    cudaError_t err = cudaDeviceReset();
    if (err != cudaSuccess) {
        std::cerr << "CUDA device reset failed: " << cudaGetErrorString(err) << "\n";
    }
}


}