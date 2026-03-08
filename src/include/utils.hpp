#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)         \
                      << " (" << err << ")"                               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)
