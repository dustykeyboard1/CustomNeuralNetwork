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

}