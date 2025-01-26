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

        // Debug: Print output for each thread
        printf("Thread %d: output[%d] = %f\n", col, col, sum);
    }
}



__global__ void initializeCurandStates(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void AddBiasKernel(const float* output, const float* bias, float* result, int batchSize, int outputSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate row and column based on idx
    int row = idx / outputSize;
    int col = idx % outputSize;

    if (row < batchSize && col < outputSize) {
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
    size_t sizeA = colsA * rowsA * sizeof(float);
    size_t sizeB = colsB * rowsB * sizeof(float);
    size_t sizeC = rowsA * colsB * sizeof(float);



    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeC, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((colsB + 15) / 16, (rowsA + 15) / 16);

    MultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, rowsB,
                                                                colsB);

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

    // Allocate GPU memory for weights
    cudaMalloc(&d_weights, size);

    // No need to copy from host weights (they're being initialized directly on GPU)
    cudaMemset(d_weights, 0, size);

    // Define grid and block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    // Convert the initialization type to an integer code
    int initTypeCode = 0; 
    if (initType == "xavier") {
        initTypeCode = 1;
    } else if (initType == "he") {
        initTypeCode = 2;
    }

    // Allocate memory for cuRAND states
    int totalSize = rows * cols;
    curandState* d_states;
    cudaMalloc(&d_states, totalSize * sizeof(curandState));

    // Initialize cuRAND states
    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    initializeCurandStates<<<gridSize, blockSize>>>(d_states, time(0));

    // Launch the kernel to initialize weights
    initializeWeightsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_weights, rows, cols, d_states, initTypeCode);
    cudaDeviceSynchronize();

    // Copy weights back to host
    cudaMemcpy(weights, d_weights, size, cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_weights);
    cudaFree(d_states);
}


void addBias(const float* output, const float* bias, float* result, int batchSize, int outputSize) {
        size_t totalElements = batchSize * outputSize;

        if (result == nullptr) {
            cudaMalloc(&result, totalElements * sizeof(float));
        }

        int blockSize = 256;
        int gridSize = (totalElements + blockSize - 1) / blockSize;

        AddBiasKernel<<<gridSize, blockSize>>>(output, bias, result, batchSize, outputSize);

        cudaDeviceSynchronize();
}

void MatrixOps::sumAcrossRows(const float* input, float* output, int rows, int cols) {
    size_t inputSize = rows * cols * sizeof(float);
    size_t outputSize = cols * sizeof(float);

    float *d_input, *d_output;

    // Debug: Log sizes
    std::cout << "Allocating GPU memory: Input size = " << inputSize 
              << ", Output size = " << outputSize << "\n";

    // Allocate GPU memory
    cudaError_t err = cudaMalloc(&d_input, inputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for input: " << cudaGetErrorString(err) << "\n";
        return;
    }

    err = cudaMalloc(&d_output, outputSize);
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed for output: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        return;
    }

    // Copy input data to GPU
    err = cudaMemcpy(d_input, input, inputSize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to device failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Debug: Print kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Launching kernel with " << blocksPerGrid 
              << " blocks and " << threadsPerBlock << " threads\n";

    // Launch kernel
    SumAcrossRowsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Copy result back to host
    err = cudaMemcpy(output, d_output, outputSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy to host failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Free GPU memory
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