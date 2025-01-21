#include "MatrixOps.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
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

__global__ void addMultiplicationKernel(const float* A, const float* B, float* C, int rowsA, int colsA, int rowsB, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * colsB + col;
    if (row < rowsA && col < colsB) {
        float sum = 0;

        for(int i = 0; i < colsA; ++i) {
            sum += A[row * colsA + i] * B[i * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

void add(const float* A, const float* B, float* C, int rows, int cols) {
    size_t size = rows * cols * sizeof(float);

    //Allocate Memory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //Copy Data from CPU to GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);

    //Configure block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

    //Launch Kernel
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);

    //Copy Result to GPU
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost); 

    //Free memory
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

    addMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rowsA, colsA, rowsB,
                                                                colsB);

    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost); 

    //Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

}