#include <cuda_runtime.h>
#include <iostream>
#include "utils.hpp"

__global__ void vectorAdd(float* a, float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    std::cout << "Vector Add Kernel Example" << std::endl;
    const int N = 1024;
    float* h_a = (float*)malloc(sizeof(float) * N);
    float* h_b = (float*)malloc(sizeof(float) * N);
    float* h_c = (float*)malloc(sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i;
    }
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_a, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void**)&d_b, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void**)&d_c, sizeof(float) * N));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, sizeof(float) * N, cudaMemcpyHostToDevice));

    dim3 block(64);
    dim3 grid((N + block.x - 1) / block.x);

    vectorAdd<<<grid, block>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        std::cout << h_c[i] << " ";
    }
}