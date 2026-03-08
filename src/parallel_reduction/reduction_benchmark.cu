/**
 * 归约核函数性能基准测试
 * 测试 3 个 shared memory 归约 kernel，输出格式对齐 NVIDIA 官方 "Performance for 4M element reduction" 表格。
 *
 * 编译后运行: ./reduction_benchmark
 */

#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// ============== 三个归约 Kernel（与 parallel_reduction.cu 中一致） ==============

/** Kernel 1: 交错寻址 + 分支发散 (interleaved addressing with divergent branching) */
__global__ void reduce_kernel_shmem_1(const float* input, float* output, int n) {
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int l_idx = threadIdx.x;
    extern __shared__ float sdata[];

    if (g_idx < n) {
        sdata[l_idx] = input[g_idx];
    } else {
        sdata[l_idx] = 0.0f;
    }
    __syncthreads();

    int block_len = blockDim.x;
    int stride = 1;
    while (stride < block_len) {
        if (l_idx % (2 * stride) == 0) {
            if (l_idx + stride >= block_len) {
                sdata[l_idx] = sdata[l_idx];
            } else {
                sdata[l_idx] = sdata[l_idx] + sdata[l_idx + stride];
            }
        }
        __syncthreads();
        stride *= 2;
    }
    if (l_idx == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/** Kernel 2: 交错寻址 + bank conflicts (interleaved addressing with bank conflicts) */
__global__ void reduce_kernel_shmem_2(const float* input, float* output, int n) {
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int l_idx = threadIdx.x;
    extern __shared__ float sdata[];

    if (g_idx < n) {
        sdata[l_idx] = input[g_idx];
    } else {
        sdata[l_idx] = 0.0f;
    }
    __syncthreads();

    int block_len = blockDim.x;
    int stride = 1;
    while (stride < block_len) {
        int p_idx = l_idx * 2 * stride;
        if (p_idx < block_len) {
            if (p_idx + stride < block_len) {
                sdata[p_idx] = sdata[p_idx] + sdata[p_idx + stride];
            }
        }
        __syncthreads();
        stride *= 2;
    }
    if (l_idx == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/** Kernel 3: 顺序寻址 (sequential addressing) */
__global__ void reduce_kernel_shmem_3(const float* input, float* output, int n) {
    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int l_idx = threadIdx.x;
    extern __shared__ float sdata[];

    if (g_idx < n) {
        sdata[l_idx] = input[g_idx];
    } else {
        sdata[l_idx] = 0.0f;
    }
    __syncthreads();

    int block_len = blockDim.x;
    int stride = block_len / 2;
    while (stride > 0) {
        if (l_idx < stride) {
            sdata[l_idx] = sdata[l_idx] + sdata[l_idx + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }
    if (l_idx == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/** Kernel 4: 每 block 处理 2*blockDim.x 个元素，每线程读 2 个，shmem 大小为 2*blockDim.x */
__global__ void reduce_kernel_shmem_4(const float* input, float* output, int n) {
    // 本 block 负责 [base, base+2*blockDim.x)，每线程 2 个
    int base = blockIdx.x * (2 * blockDim.x);
    int g_idx = base + threadIdx.x;
    int l_idx = threadIdx.x;
    extern __shared__ float sdata[];

    if (g_idx < n) {
        sdata[l_idx] = input[g_idx];
    } else {
        sdata[l_idx] = 0.0f;
    }
    int g_idx2 = base + blockDim.x + threadIdx.x;
    if (g_idx2 < n) {
        sdata[l_idx + blockDim.x] = input[g_idx2];
    } else {
        sdata[l_idx + blockDim.x] = 0.0f;
    }
    __syncthreads();

    int block_len = 2 * blockDim.x;  // shmem 有效长度
    int stride = block_len / 2;
    while (stride > 0) {
        if (l_idx < stride) {
            sdata[l_idx] = sdata[l_idx] + sdata[l_idx + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }
    if (l_idx == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__device__ void warp_reduce(volatile float* sdata, int l_idx) {
    sdata[l_idx] = sdata[l_idx] + sdata[l_idx + 32];
    sdata[l_idx] = sdata[l_idx] + sdata[l_idx + 16];
    sdata[l_idx] = sdata[l_idx] + sdata[l_idx + 8];
    sdata[l_idx] = sdata[l_idx] + sdata[l_idx + 4];
    sdata[l_idx] = sdata[l_idx] + sdata[l_idx + 2];
    sdata[l_idx] = sdata[l_idx] + sdata[l_idx + 1];
}

/** Kernel 5: 每 block 处理 2*blockDim.x 个元素，每线程读 2 个，shmem 大小为 2*blockDim.x */
__global__ void reduce_kernel_shmem_5(const float* input, float* output, int n) {
    // 本 block 负责 [base, base+2*blockDim.x)，每线程 2 个
    int base = blockIdx.x * (2 * blockDim.x);
    int g_idx = base + threadIdx.x;
    int l_idx = threadIdx.x;
    extern __shared__ float sdata[];

    if (g_idx < n) {
        sdata[l_idx] = input[g_idx];
    } else {
        sdata[l_idx] = 0.0f;
    }
    int g_idx2 = base + blockDim.x + threadIdx.x;
    if (g_idx2 < n) {
        sdata[l_idx + blockDim.x] = input[g_idx2];
    } else {
        sdata[l_idx + blockDim.x] = 0.0f;
    }
    __syncthreads();

    int block_len = 2 * blockDim.x;  // shmem 有效长度
    int stride = block_len / 2;
    while (stride > 32) {
        if (l_idx < stride) {
            sdata[l_idx] = sdata[l_idx] + sdata[l_idx + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }
    if (l_idx < 32) {
        warp_reduce(sdata, l_idx);
    }

    if (l_idx == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
/** Kernel 5: 每 block 处理 2*blockDim.x 个元素，每线程读 2 个，shmem 大小为 2*blockDim.x */
__global__ void reduce_kernel_shmem_6(const float* input, float* output, int n) {
    // 本 block 负责 [base, base+2*blockDim.x)，每线程 2 个
    int base = blockIdx.x * (2 * blockDim.x);
    int g_idx = base + threadIdx.x;
    int l_idx = threadIdx.x;
    extern __shared__ float sdata[];

    if (g_idx < n) {
        sdata[l_idx] = input[g_idx];
    } else {
        sdata[l_idx] = 0.0f;
    }
    int g_idx2 = base + blockDim.x + threadIdx.x;
    if (g_idx2 < n) {
        sdata[l_idx + blockDim.x] = input[g_idx2];
    } else {
        sdata[l_idx + blockDim.x] = 0.0f;
    }
    __syncthreads();

    int block_len = 2048;  // shmem 有效长度
    int stride = block_len / 2;
    #pragma unroll
    while (stride > 32) {
        if (l_idx < stride) {
            sdata[l_idx] = sdata[l_idx] + sdata[l_idx + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }
    if (l_idx < 32) {
        warp_reduce(sdata, l_idx);
    }

    if (l_idx == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/** Kernel 5: 每 block 处理 2*blockDim.x 个元素，每线程读 2 个的时候，直接进行add，shmem 大小为 1*blockDim.x */
__global__ void reduce_kernel_shmem_7(const float* input, float* output, int n) {
    // 本 block 负责 [base, base+2*blockDim.x)，每线程 2 个
    int base = blockIdx.x * (2 * blockDim.x);
    int g_idx = base + threadIdx.x;
    int l_idx = threadIdx.x;
    extern __shared__ float sdata[];

    if (g_idx < n) {
        sdata[l_idx] = input[g_idx];
    } else {
        sdata[l_idx] = 0.0f;
    }
    int g_idx2 = base + blockDim.x + threadIdx.x;
    if (g_idx2 < n) {
        sdata[l_idx] += input[g_idx2];
    } 
    __syncthreads();

    int block_len = 1024;  // shmem 有效长度
    int stride = block_len / 2;
    #pragma unroll
    while (stride > 32) {
        if (l_idx < stride) {
            sdata[l_idx] = sdata[l_idx] + sdata[l_idx + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }
    if (l_idx < 32) {
        warp_reduce(sdata, l_idx);
    }

    if (l_idx == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/** Kernel 5: 每 block 处理 2*blockDim.x 个元素，每线程读 2 个的时候，直接进行add，shmem 大小为 1*blockDim.x */
__global__ void reduce_kernel_shmem_8(const float* input, float* output, int n) {
    // 本 block 负责 [base, base+2*blockDim.x)，每线程 2 个
    int grid_size = gridDim.x * blockDim.x * 2;
    int base = blockIdx.x * (2 * blockDim.x);
    int g_idx = base + threadIdx.x;
    int l_idx = threadIdx.x;
    extern __shared__ float sdata[];

    sdata[l_idx] = 0.0f;  // shared memory 未初始化，必须显式置零
    while(g_idx < n) {
        sdata[l_idx] += (input[g_idx] + input[g_idx + blockDim.x]);
        g_idx += grid_size;
    }
    __syncthreads();

    int block_len = 1024;  // shmem 有效长度
    int stride = block_len / 2;
    #pragma unroll
    while (stride > 32) {
        if (l_idx < stride) {
            sdata[l_idx] = sdata[l_idx] + sdata[l_idx + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }
    if (l_idx < 32) {
        warp_reduce(sdata, l_idx);
    }

    if (l_idx == 0) {
        output[blockIdx.x] = sdata[0];
    }
}


// ============== CPU 参考实现 ==============

/** CPU 顺序求和，用 Kahan 补偿算法减少浮点累积误差 */
float cpu_reduce_ref(const float* input, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += static_cast<double>(input[i]);
    return static_cast<float>(sum);
}

// ============== 正确性验证 ==============

/** 运行 kernel（1x block 配置）并返回归约结果 */
template<void (*Kernel)(const float*, float*, int)>
float run_kernel_1x(const float* h_input, int n) {
    const int block_size = 1024;
    dim3 block(block_size);
    dim3 grid((n + block_size - 1) / block_size);
    size_t shmem = sizeof(float) * block_size;

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(float) * grid.x);
    cudaMemcpy(d_input, h_input, sizeof(float) * n, cudaMemcpyHostToDevice);

    Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    std::vector<float> h_output(grid.x);
    cudaMemcpy(h_output.data(), d_output, sizeof(float) * grid.x, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    float total = 0.0f;
    for (int i = 0; i < static_cast<int>(grid.x); i++) total += h_output[i];
    return total;
}

/** 运行 kernel（2x block 配置）并返回归约结果 */
template<void (*Kernel)(const float*, float*, int)>
float run_kernel_2x(const float* h_input, int n) {
    const int elems_per_block = 2048;
    dim3 block(1024);
    dim3 grid((n + elems_per_block - 1) / elems_per_block);
    size_t shmem = sizeof(float) * elems_per_block;

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(float) * grid.x);
    cudaMemcpy(d_input, h_input, sizeof(float) * n, cudaMemcpyHostToDevice);

    Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    std::vector<float> h_output(grid.x);
    cudaMemcpy(h_output.data(), d_output, sizeof(float) * grid.x, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    float total = 0.0f;
    for (int i = 0; i < static_cast<int>(grid.x); i++) total += h_output[i];
    return total;
}

/** 运行 kernel 7（每 block 处理 2*blockDim.x 个元素，shmem = 1*blockDim.x floats）并返回归约结果 */
template<void (*Kernel)(const float*, float*, int)>
float run_kernel_7x(const float* h_input, int n) {
    const int elems_per_block = 2048;
    dim3 block(1024);
    dim3 grid((n + elems_per_block - 1) / elems_per_block);
    size_t shmem = sizeof(float) * 1024;  // 加载时已完成首次加法，shmem 仅需 1x

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(float) * grid.x);
    cudaMemcpy(d_input, h_input, sizeof(float) * n, cudaMemcpyHostToDevice);

    Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    std::vector<float> h_output(grid.x);
    cudaMemcpy(h_output.data(), d_output, sizeof(float) * grid.x, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    float total = 0.0f;
    for (int i = 0; i < static_cast<int>(grid.x); i++) total += h_output[i];
    return total;
}

/** 运行 kernel 8（grid-stride loop，grid = kernel7 的 1/4，每 block 处理 8*blockDim.x 个元素，shmem = 1*blockDim.x floats）并返回归约结果 */
template<void (*Kernel)(const float*, float*, int)>
float run_kernel_8x(const float* h_input, int n) {
    const int elems_per_block = 22 * 1024;  // grid 缩小为 1/4，每 block 负责 4 倍的元素
    dim3 block(1024);
    dim3 grid((n + elems_per_block - 1) / elems_per_block);
    size_t shmem = sizeof(float) * 1024;

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(float) * grid.x);
    cudaMemcpy(d_input, h_input, sizeof(float) * n, cudaMemcpyHostToDevice);

    Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    std::vector<float> h_output(grid.x);
    cudaMemcpy(h_output.data(), d_output, sizeof(float) * grid.x, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    float total = 0.0f;
    for (int i = 0; i < static_cast<int>(grid.x); i++) total += h_output[i];
    return total;
}

void print_correctness(const char* name, float got, float ref) {
    float rel_err = std::fabs(got - ref) / (std::fabs(ref) + 1e-6f);
    bool pass = rel_err < 1e-3f;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]")
              << "  " << std::left << std::setw(52) << name
              << "  got=" << std::fixed << std::setprecision(2) << got
              << "  ref=" << ref
              << "  rel_err=" << std::scientific << std::setprecision(2) << rel_err
              << "\n";
}

// ============== 基准测试逻辑 ==============

constexpr int BLOCK_SIZE = 1024;
constexpr int N_ELEMENTS = 1 << 22;  // 4M elements，与 NVIDIA 表格一致
constexpr int WARMUP_RUNS = 3;
constexpr int TIMED_RUNS = 20;

/** 只测量单次 kernel 执行时间（不含 H2D/D2H），返回平均时间 ms。每 block 处理 BLOCK_SIZE 个元素。 */
template<void (*Kernel)(const float*, float*, int)>
double measure_kernel_time_ms(int n) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    size_t shmem = sizeof(float) * BLOCK_SIZE;

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(float) * grid.x);

    std::vector<float> h_input(n, 1.0f);
    cudaMemcpy(d_input, h_input.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    for (int i = 0; i < WARMUP_RUNS; i++) {
        Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_RUNS; i++) {
        Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = static_cast<double>(total_ms) / TIMED_RUNS;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return avg_ms;
}

/** Kernel 4/5 专用：每 block 处理 2*BLOCK_SIZE 个元素，shmem = 2*BLOCK_SIZE floats */
template<void (*Kernel)(const float*, float*, int)>
double measure_kernel_2x_time_ms(int n) {
    const int elems_per_block = 2 * BLOCK_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + elems_per_block - 1) / elems_per_block);
    size_t shmem = sizeof(float) * elems_per_block;

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(float) * grid.x);

    std::vector<float> h_input(n, 1.0f);
    cudaMemcpy(d_input, h_input.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    for (int i = 0; i < WARMUP_RUNS; i++) {
        Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_RUNS; i++) {
        Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = static_cast<double>(total_ms) / TIMED_RUNS;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return avg_ms;
}

/** Kernel 7 专用：每 block 处理 2*BLOCK_SIZE 个元素，shmem = 1*BLOCK_SIZE floats（加载时完成首次加法） */
template<void (*Kernel)(const float*, float*, int)>
double measure_kernel_7_time_ms(int n) {
    const int elems_per_block = 2 * BLOCK_SIZE;
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + elems_per_block - 1) / elems_per_block);
    size_t shmem = sizeof(float) * BLOCK_SIZE;  // 仅需 1x shmem

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(float) * grid.x);

    std::vector<float> h_input(n, 1.0f);
    cudaMemcpy(d_input, h_input.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    for (int i = 0; i < WARMUP_RUNS; i++) {
        Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_RUNS; i++) {
        Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = static_cast<double>(total_ms) / TIMED_RUNS;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return avg_ms;
}

/** Kernel 8 专用：grid-stride loop，grid = 1/4 of kernel7，shmem = 1*BLOCK_SIZE floats */
template<void (*Kernel)(const float*, float*, int)>
double measure_kernel_8_time_ms(int n) {
    const int elems_per_block = 8 * BLOCK_SIZE;  // 每 block 负责 4 倍的元素量
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + elems_per_block - 1) / elems_per_block);
    size_t shmem = sizeof(float) * BLOCK_SIZE;

    float *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, sizeof(float) * n);
    cudaMalloc(&d_output, sizeof(float) * grid.x);

    std::vector<float> h_input(n, 1.0f);
    cudaMemcpy(d_input, h_input.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    for (int i = 0; i < WARMUP_RUNS; i++) {
        Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < TIMED_RUNS; i++) {
        Kernel<<<grid, block, shmem>>>(d_input, d_output, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start, stop);
    double avg_ms = static_cast<double>(total_ms) / TIMED_RUNS;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);

    return avg_ms;
}

/** 带宽 (GB/s)：归约主要读入 n 个 float */
inline double bandwidth_gbps(int n, double time_ms) {
    if (time_ms <= 0) return 0.0;
    double bytes = static_cast<double>(n) * sizeof(float);
    double time_sec = time_ms * 1e-3;
    return (bytes / time_sec) / 1e9;
}

// 表格列宽，保证表头与数据列对齐
constexpr int W_KERNEL = 52;
constexpr int W_TIME   = 12;
constexpr int W_BW     = 12;
constexpr int W_STEP   = 14;
constexpr int W_CUM    = 20;
constexpr int W_TOTAL  = W_KERNEL + W_TIME + W_BW + W_STEP + 1 + W_CUM;

void print_separator() {
    std::cout << std::string(W_TOTAL, '-') << "\n";
}

int main(int argc, char** argv) {
    int n = N_ELEMENTS;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }

    // 生成随机输入（用小值避免浮点溢出）
    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) h_input[i] = static_cast<float>(rand() % 100) * 0.01f;

    // ---- 正确性验证 ----
    float ref = cpu_reduce_ref(h_input.data(), n);
    std::cout << "\n  Correctness check (n=" << n << ", ref=" << std::fixed << std::setprecision(2) << ref << ")\n";
    std::cout << std::string(W_TOTAL, '-') << "\n";
    print_correctness("Kernel 1: divergent branching",    run_kernel_1x<reduce_kernel_shmem_1>(h_input.data(), n), ref);
    print_correctness("Kernel 2: bank conflicts",         run_kernel_1x<reduce_kernel_shmem_2>(h_input.data(), n), ref);
    print_correctness("Kernel 3: sequential addressing",  run_kernel_1x<reduce_kernel_shmem_3>(h_input.data(), n), ref);
    print_correctness("Kernel 4: 2x shmem, sequential",  run_kernel_2x<reduce_kernel_shmem_4>(h_input.data(), n), ref);
    print_correctness("Kernel 5: 2x shmem, warp unroll", run_kernel_2x<reduce_kernel_shmem_5>(h_input.data(), n), ref);
    print_correctness("Kernel 6: 2x shmem, loop unroll", run_kernel_2x<reduce_kernel_shmem_6>(h_input.data(), n), ref);
    print_correctness("Kernel 7: 2x elem/block, add-on-load, 1x shmem", run_kernel_7x<reduce_kernel_shmem_7>(h_input.data(), n), ref);
    print_correctness("Kernel 8: grid-stride loop, 1/4 grid, 1x shmem", run_kernel_8x<reduce_kernel_shmem_8>(h_input.data(), n), ref);
    std::cout << std::string(W_TOTAL, '-') << "\n";

    // ---- 性能 benchmark ----
    std::cout << "\n  Performance for " << (n >> 20) << "M element reduction (n = " << n << ")\n\n";

    double t1 = measure_kernel_time_ms<reduce_kernel_shmem_1>(n);
    double t2 = measure_kernel_time_ms<reduce_kernel_shmem_2>(n);
    double t3 = measure_kernel_time_ms<reduce_kernel_shmem_3>(n);
    double t4 = measure_kernel_2x_time_ms<reduce_kernel_shmem_4>(n);
    double t5 = measure_kernel_2x_time_ms<reduce_kernel_shmem_5>(n);
    double t6 = measure_kernel_2x_time_ms<reduce_kernel_shmem_6>(n);
    double t7 = measure_kernel_7_time_ms<reduce_kernel_shmem_7>(n);
    double t8 = measure_kernel_8_time_ms<reduce_kernel_shmem_8>(n);

    double bw1 = bandwidth_gbps(n, t1);
    double bw2 = bandwidth_gbps(n, t2);
    double bw3 = bandwidth_gbps(n, t3);
    double bw4 = bandwidth_gbps(n, t4);
    double bw5 = bandwidth_gbps(n, t5);
    double bw6 = bandwidth_gbps(n, t6);
    double bw7 = bandwidth_gbps(n, t7);
    double bw8 = bandwidth_gbps(n, t8);

    double step2 = t1 / t2;
    double step3 = t2 / t3;
    double step4 = t3 / t4;
    double step5 = t4 / t5;
    double step6 = t5 / t6;
    double step7 = t6 / t7;
    double step8 = t7 / t8;
    double cum2 = t1 / t2;
    double cum3 = t1 / t3;
    double cum4 = t1 / t4;
    double cum5 = t1 / t5;
    double cum6 = t1 / t6;
    double cum7 = t1 / t7;
    double cum8 = t1 / t8;

    // 表头（两行）
    print_separator();
    std::cout << std::left  << std::setw(W_KERNEL) << "Kernel"
              << std::right << std::setw(W_TIME)   << "Time (ms)"
              << std::setw(W_BW)   << "Bandwidth"
              << std::setw(W_STEP) << "Step"
              << " "
              << std::setw(W_CUM)  << "Cumulative"
              << "\n";
    std::cout << std::left  << std::setw(W_KERNEL) << ""
              << std::right << std::setw(W_TIME)   << "(2^22 flt)"
              << std::setw(W_BW)   << "(GB/s)"
              << std::setw(W_STEP) << "Speedup"
              << " "
              << std::setw(W_CUM)  << "Speedup"
              << "\n";
    print_separator();

    auto row = [](const char* name, double time_ms, double gbps,
                  bool has_step, double step_val, bool has_cum, double cum_val) {
        std::cout << std::left  << std::setw(W_KERNEL) << name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(W_TIME) << time_ms
                  << std::setw(W_BW)   << gbps
                  << std::setw(W_STEP);
        if (has_step) std::cout << std::setprecision(2) << step_val << "x"; else std::cout << "—";
        std::cout << " " << std::setw(W_CUM);
        if (has_cum) std::cout << std::setprecision(2) << cum_val << "x"; else std::cout << "—";
        std::cout << "\n";
    };

    row("Kernel 1: interleaved addressing with divergent branching", t1, bw1, false, 0, false, 0);
    row("Kernel 2: interleaved addressing with bank conflicts",      t2, bw2, true, step2, true, cum2);
    row("Kernel 3: sequential addressing",                           t3, bw3, true, step3, true, cum3);
    row("Kernel 4: 2*block elements, sequential (2x shmem)",         t4, bw4, true, step4, true, cum4);
    row("Kernel 5: 2*block elements, warp unrolled  (2x shmem)",     t5, bw5, true, step5, true, cum5);
    row("Kernel 6: 2*block elements, loop unrolled  (2x shmem)",     t6, bw6, true, step6, true, cum6);
    row("Kernel 7: 2*block elements, add-on-load    (1x shmem)",     t7, bw7, true, step7, true, cum7);
    row("Kernel 8: grid-stride loop, 1/4 grid      (1x shmem)",     t8, bw8, true, step8, true, cum8);

    print_separator();
    std::cout << "\n  (Kernel time only; " << TIMED_RUNS << " runs averaged, " << n << " elements)\n\n";

    return 0;
}
