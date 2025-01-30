#include "LossOps.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <iostream>

namespace LossOps {
__global__ void MSEKernel(const float* yTrue, const float* yPred, float* loss, int size) { 
    extern __shared__ float sharedLoss[];  
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedLoss[threadIdx.x] = 0.0f;

    if (idx < size) {
        float diff = yTrue[idx] - yPred[idx];
        sharedLoss[threadIdx.x] = diff * diff;
    }

    __syncthreads();

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

float LossOps::MeanSquaredError(const float* targets, const float* predictions, int size, bool isGPU) {
    float *d_targets, *d_predictions, *d_loss;
    float totalLoss = 0.0f;

    if (isGPU) {
        // Data is already on GPU, just allocate loss buffer
        cudaMalloc(&d_loss, size * sizeof(float));
        
        // Launch kernel directly with GPU data
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        MSEKernel<<<numBlocks, blockSize>>>(targets, predictions, d_loss, size);
    } else {
        // Allocate and copy data to GPU
        cudaMalloc(&d_targets, size * sizeof(float));
        cudaMalloc(&d_predictions, size * sizeof(float));
        cudaMalloc(&d_loss, size * sizeof(float));

        cudaMemcpy(d_targets, targets, size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_predictions, predictions, size * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        MSEKernel<<<numBlocks, blockSize>>>(d_targets, d_predictions, d_loss, size);
    }

    // Reduce on GPU to compute sum
    // ... (reduction kernel to sum all losses)
    
    // Copy final result back
    float meanLoss;
    cudaMemcpy(&meanLoss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    if (!isGPU) {
        cudaFree(d_targets);
        cudaFree(d_predictions);
    }
    cudaFree(d_loss);

    return meanLoss / size;
}

float LossOps::gpuCrossEntropyLoss(const float* yTrue, const float* yPred, int batchSize, int numClasses) {
    float* d_yTrue;
    float* d_yPred;
    float* d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_yTrue, sizeof(float) * batchSize * numClasses);
    cudaMalloc(&d_yPred, sizeof(float) * batchSize * numClasses);
    cudaMalloc(&d_loss, sizeof(float));

    cudaMemcpy(d_yTrue, yTrue, sizeof(float) * batchSize * numClasses, cudaMemcpyHostToDevice);
    cudaMemcpy(d_yPred, yPred, sizeof(float) * batchSize * numClasses, cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (batchSize * numClasses + threadsPerBlock - 1) / threadsPerBlock;

    crossEntropyLossKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_yTrue, d_yPred, d_loss, numClasses, batchSize);

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_yTrue);
    cudaFree(d_yPred);
    cudaFree(d_loss);

    return h_loss / batchSize;
}


}