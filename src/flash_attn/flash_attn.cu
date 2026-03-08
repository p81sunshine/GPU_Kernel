#include <__clang_cuda_builtin_vars.h>
#include <cuda_runtime.h>
#include <iostream>
#include "utils.hpp"

void init_qkv(float* Q, float* K, float* V, int B, int L, int H, int D, int seed) {
    srand(seed);
    for (int i = 0; i < B * L * H * D; i++) {
        Q[i] = rand() / (float)RAND_MAX;
        K[i] = rand() / (float)RAND_MAX;
        V[i] = rand() / (float)RAND_MAX;
    }
}

void print_matrix(float* matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << matrix[i * N + j] << " ";
        }
        std::cout << "========" << std::endl;
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

__global__ void self_attention(float* Q, float* K, float* V, int B, int L, int H, int D) {
    // first step: QK^T,
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    // (B,H,L,D)
    int batch_idx = row / (L * H);
    int head_idx = row % (L * H)
    
}

int main(int argc, char** argv) {
    std::cout << "Flash Attention V1" << std::endl;
    int B,L,H,D;
    B = 1;
    L = 10;
    H = 2;
    D = 4;
    float *Q, *K, *V;
    Q = (float*)malloc(sizeof(float) * B * L * H * D);
    K = (float*)malloc(sizeof(float) * B * L * H * D);
    V = (float*)malloc(sizeof(float) * B * L * H * D);

    init_qkv(Q, K, V, B, L, H, D, 123);
    // print_matrix(Q, B, L * H * D);


    float* Qd, *Kd, *Vd;
    CUDA_CHECK(cudaMalloc(&Qd, sizeof(float) * B * L * H * D));
    CUDA_CHECK(cudaMalloc(&Kd, sizeof(float) * B * L * H * D));
    CUDA_CHECK(cudaMalloc(&Vd, sizeof(float) * B * L * H * D));
    CUDA_CHECK(cudaMemcpy(Qd, Q, sizeof(float) * B * L * H * D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Kd, K, sizeof(float) * B * L * H * D, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Vd, V, sizeof(float) * B * L * H * D, cudaMemcpyHostToDevice));


    return 0;
}