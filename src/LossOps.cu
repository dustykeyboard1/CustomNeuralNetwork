#include "LossOps.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <iostream>

namespace LossOps {

// CUDA Kernels for Loss Computation
__global__ void MSEKernel(const float* yTrue, const float* yPred, float* loss, int size) { 
    extern __shared__ float sharedLoss[];  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedLoss[threadIdx.x] = 0.0f;

    if (idx < size) {
        float diff = yTrue[idx] - yPred[idx];
        sharedLoss[threadIdx.x] = diff * diff;
    }

    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /=2) {
        if (threadIdx.x < stride) {
            sharedLoss[threadIdx.x] += sharedLoss[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(loss, sharedLoss[0]);
    }
}

// Kernel for computing error gradients
__global__ void computeErrorKernel(const float* yTrue, const float* yPred, float* error, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = yPred[idx] - yTrue[idx];
    }
}

// Kernel for cross entropy loss computation
__global__ void crossEntropyLossKernel(const float* yTrue, const float* yPred, float* loss, int numClasses, int batchSize) { 
    extern __shared__ float sharedLoss[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedLoss[threadIdx.x] = 0.0f;

    if (idx < batchSize*numClasses) {
        if (yTrue[idx] > 0.0f) {
            sharedLoss[threadIdx.x] = -yTrue[idx] * logf(yPred[idx] + 1e-8);
        }
    }
    __syncthreads();

    // Parallel reduction for loss computation
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            sharedLoss[threadIdx.x] += sharedLoss[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        atomicAdd(loss, sharedLoss[0]);
    }
}

// Mean Squared Error loss computation
float LossOps::MeanSquaredError(const float* targets, const float* predictions, int size, bool isGPU) {
    float *d_targets, *d_predictions, *d_loss;
     
    if (isGPU) {
        // Use GPU data directly
        cudaMalloc(&d_loss, size * sizeof(float));
        
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        MSEKernel<<<numBlocks, blockSize>>>(targets, predictions, d_loss, size);
    } else {
        // Copy data to GPU
        cudaMalloc(&d_targets, size * sizeof(float));
        cudaMalloc(&d_predictions, size * sizeof(float));
        cudaMalloc(&d_loss, size * sizeof(float));

        cudaMemcpy(d_targets, targets, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_predictions, predictions, size * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        MSEKernel<<<numBlocks, blockSize>>>(d_targets, d_predictions, d_loss, size);
    }

    // Get final loss value
    float totalLoss;
    cudaMemcpy(&totalLoss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup GPU memory
    if (!isGPU) {
        cudaFree(d_targets);
        cudaFree(d_predictions);
    }
    cudaFree(d_loss);
    return totalLoss;
}

// Cross Entropy loss computation
float LossOps::gpuCrossEntropyLoss(const float* yTrue, const float* yPred, int batchSize, int numClasses) {
    float* d_yTrue;
    float* d_yPred;
    float* d_loss;
    float h_loss = 0.0f;

    // Allocate and copy data to GPU
    cudaMalloc(&d_yTrue, sizeof(float) * batchSize * numClasses);
    cudaMalloc(&d_yPred, sizeof(float) * batchSize * numClasses);
    cudaMalloc(&d_loss, sizeof(float));

    cudaMemcpy(d_yTrue, yTrue, sizeof(float) * batchSize * numClasses, cudaMemcpyHostToDevice);
    cudaMemcpy(d_yPred, yPred, sizeof(float) * batchSize * numClasses, cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize * numClasses + threadsPerBlock - 1) / threadsPerBlock;
    crossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_yTrue, d_yPred, d_loss, numClasses, batchSize);

    // Get result and cleanup
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_yTrue);
    cudaFree(d_yPred);
    cudaFree(d_loss);

    return h_loss / batchSize;
}

// Compute error gradients for backpropagation
void LossOps::computeError(const float* yTrue, const float* yPred, float* error, int size, bool isGPU) {
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);

    if (isGPU) {
        // Use GPU data directly
        computeErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(yTrue, yPred, error, size);
    } else {
        // Copy data to GPU
        float *d_yTrue, *d_yPred;
        cudaMalloc(&d_yTrue, size * sizeof(float));
        cudaMalloc(&d_yPred, size * sizeof(float));
        
        cudaMemcpy(d_yTrue, yTrue, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yPred, yPred, size * sizeof(float), cudaMemcpyHostToDevice);
        
        computeErrorKernel<<<blocksPerGrid, threadsPerBlock>>>(d_yTrue, d_yPred, error, size);
        
        cudaFree(d_yTrue);
        cudaFree(d_yPred);
    }
    
    cudaDeviceSynchronize();
}
}