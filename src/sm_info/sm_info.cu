#include <cuda_runtime.h>
#include <iostream>
#include "utils.hpp"

int main(int argc, char** argv) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;

    cudaDeviceProp prop;
    int dev = 0;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Device " << dev << ": " << prop.name << std::endl;

    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << (prop.totalGlobalMem >> 20) << " MB" << std::endl;
    std::cout << "  Shared Memory per Block: " << (prop.sharedMemPerBlock >> 10) << " KB" << std::endl;
    std::cout << "  maxBlocksPerMultiProcessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "  sharedMemPerMultiprocessor: " << (prop.sharedMemPerMultiprocessor >> 10) << " KB" << std::endl;
    std::cout << "  Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads Dim: (" 
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max Grid Size: (" 
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "  Memory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
    std::cout << "  Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
    std::cout << "  MultiProcessor Count: " << prop.multiProcessorCount << std::endl;
    return 0;
}