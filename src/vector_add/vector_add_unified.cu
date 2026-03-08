#include<cuda_runtime.h>
#include<iostream>

__global__ void vectorAdd(float* a, float* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    std::cout << "Vector Add Kernel Example" << std::endl;
    const int N = 1024;
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    cudaMallocManaged(&a, sizeof(float) * N);
    cudaMallocManaged(&b, sizeof(float) * N);
    cudaMallocManaged(&c, sizeof(float) * N);
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i;
    }
    vectorAdd<<<1, 1024>>>(a, b, c, N);
    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++) {
        std::cout << c[i] << " ";
    }
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}