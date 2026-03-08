#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <ostream>
#include "utils.hpp"

void show_content(float* a, int M, int N) {
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++) {
            std::cout << a[i*N+j] << " ";
        }
        std::cout << "\n" << std::endl;
    }
}

void transpose(float* a,float* c, int M, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            c[i*M+j] = a[j*N+i];
        }
    }
}

__global__ void transpose_kernel(float* a, float* c, int M, int N) {
    // Assume 2d block
    // Assume a is original matrix, its shape is (M,N)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx < M && idy < N) {
        c[idy*M+idx] = a[idx*N+idy];
    }
}

int main(int argc, char** argv) {
    const int M = 5;
    const int N = 10;
    float* h_a = (float*)malloc(sizeof(float) * M * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            h_a[i * N + j] = j;
        }
    }
    float* h_c = (float*)malloc(sizeof(float) * M * N);
    show_content(h_a, M,  N);
    transpose(h_a,h_c, M,  N);
    std::cout << "Transpose:" << "\n";
    show_content(h_c, N,  M);

    float* d_a = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a,sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc(&d_c,sizeof(float) * M * N));

    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, d_a));
    std::cout << "Memory type: " << attr.type << " (0=unregistered, 1=host, 2=device, 3=managed)" << "\n";
    
    std::cout << "Original matrix:" << "\n";
    show_content(h_a, M, N);
    cudaMemcpy(d_a,h_a,sizeof(float) * M * N,cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    transpose_kernel<<<grid,block>>>(d_a,d_c,M,N);
    
    cudaDeviceSynchronize();
    cudaMemcpy(h_c,d_c,sizeof(float) * M * N,cudaMemcpyDeviceToHost);
    std::cout << "Transpose with kernel:" << "\n";
    show_content(h_c,N,M);

    return 0;
}