#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

__global__ void reduce_kernel(float* input, float* output, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if((2*idx + 1) == n) {
        output[idx] = input[2 * idx];
    } else if (2*idx < n) {
        output[idx] = input[2 * idx] + input[2 * idx + 1];
    }
}

__global__ void reduce_kernel_shmem_3(const float* input, float* output, int n) {
    int g_idx = blockDim.x * blockIdx.x+ threadIdx.x;
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
    while(stride > 0) {
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


__global__ void reduce_kernel_shmem_2(const float* input, float* output, int n) {
    int g_idx = blockDim.x * blockIdx.x+ threadIdx.x;
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
    while(stride < block_len) {
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

__global__ void reduce_kernel_shmem(const float* input, float* output, int n) {
    int g_idx = blockDim.x * blockIdx.x+ threadIdx.x;
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
    while(stride < block_len) {
        if (l_idx % 2*stride == 0) {
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


// Kernel 类型：(const float*, float*, int) -> void
template<void (*ReduceKernel)(const float*, float*, int)>
float reduce_in_device(float* input, int n) {
    const int BLOCK_SIZE = 1024;
    dim3 block(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float* output = new float[grid.x];
    float *input_d = nullptr;
    float *output_d = nullptr;

    cudaMalloc(&input_d, sizeof(float) * n);
    cudaMalloc(&output_d, sizeof(float) * grid.x);

    cudaMemcpy(input_d, input, sizeof(float) * n, cudaMemcpyHostToDevice);
    ReduceKernel<<<grid, block, sizeof(float) * BLOCK_SIZE>>>(input_d, output_d, n);
    cudaMemcpy(output, output_d, sizeof(float) * grid.x, cudaMemcpyDeviceToHost);
    float total = 0;
    for (int i = 0; i < grid.x; i++) {
        total += output[i];
    }

    cudaFree(input_d);
    cudaFree(output_d);
    delete[] output;
    return total;
}
// idx | 0 | 1 | 2 |
// n = 3
// idx == 0, 2*idx=0 < n = 3 (0,1)
// idx == 1, 2*idx == 2, n = 3

float reduce(float* input, int n) {
    float* input_h = (float*)malloc(n * sizeof(float));
    memcpy(input_h, input, n * sizeof(float));
    float* input_d;
    float* output_d;

    const int BLOCK_SIZE = 64;
    int input_length = n;
    while (input_length > 1) {
        int output_length = input_length % 2 == 0 ? input_length / 2 : input_length / 2 + 1;
        cudaMalloc(&input_d, sizeof(float) * input_length);
        cudaMalloc(&output_d, sizeof(float) * output_length);
        cudaMemcpy(input_d,input_h,input_length * sizeof(float), cudaMemcpyHostToDevice );
        int num_of_block = (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce_kernel<<<num_of_block,BLOCK_SIZE>>>(input_d, output_d, input_length);
        cudaMemcpy(input_h,output_d,sizeof(float) * output_length, cudaMemcpyDeviceToHost);
        cudaFree(input_d);
        cudaFree(output_d);
        input_length = output_length;
    }
    float result = input_h[0];
    free(input_h);
    return result;
}

float reduce_pingpong(float* input, int n) {
    // 1. 一次性分配两块显存，大小都取 n（最坏情况）
    float* buf[2];
    cudaMalloc(&buf[0], sizeof(float) * n);
    cudaMalloc(&buf[1], sizeof(float) * n);

    // 2. 数据只上传一次
    cudaMemcpy(buf[0], input, sizeof(float) * n, cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 64;
    int input_length = n;
    int cur = 0;  // 当前输入在 buf[cur]，输出在 buf[1-cur]

    // 3. 所有轮次都在 GPU 显存内完成，Host 不介入
    while (input_length > 1) {
        int output_length = (input_length + 1) / 2;
        int num_of_block = (output_length + BLOCK_SIZE - 1) / BLOCK_SIZE;

        reduce_kernel<<<num_of_block, BLOCK_SIZE>>>(buf[cur], buf[1 - cur], input_length);

        cur = 1 - cur;        // 交换角色：上一轮的输出变成下一轮的输入
        input_length = output_length;
    }

    // 4. 只取回最终 1 个 float
    float result;
    cudaMemcpy(&result, buf[cur], sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(buf[0]);
    cudaFree(buf[1]);
    return result;
}

/** CPU 顺序求和（与树形结果可能有浮点误差） */
static float reduce_reference_linear(const float* input, int n) {
    float sum = 0.f;
    for (int i = 0; i < n; i++) sum += input[i];
    return sum;
}

/** CPU 树形归约：与 GPU kernel 相同的两两配对方式，保证与 GPU 结果逐位一致 */
static float reduce_reference_tree(const float* input, int n) {
    if (n <= 0) return 0.f;
    std::vector<float> buf(input, input + n);
    int len = n;
    while (len > 1) {
        int out_len = (len % 2 == 0) ? len / 2 : len / 2 + 1;
        for (int idx = 0; idx < out_len; idx++) {
            if (2 * idx + 1 < len)
                buf[idx] = buf[2 * idx] + buf[2 * idx + 1];
            else  // 2*idx + 1 == len，奇数个时最后一个单独保留
                buf[idx] = buf[2 * idx];
        }
        len = out_len;
    }
    return buf[0];
}

/** Fuzzing：随机 n、随机数据，所有实现与 CPU ref 比较正确性，并统计各自耗时；in-device 同时跑 shmem 与 shmem_2 */
static bool run_fuzz(int iterations, int max_n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist_n(1, max_n);
    std::uniform_real_distribution<float> dist_val(-1e3f, 1e3f);

    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double, std::milli>;
    double total_naive_ms = 0.;
    double total_pingpong_ms = 0.;
    double total_shmem_ms = 0.;
    double total_shmem2_ms = 0.;
    double total_cpu_ms = 0.;
    double total_shmem3_ms = 0.;
    int failed_naive = 0, failed_pingpong = 0, failed_shmem = 0, failed_shmem2 = 0, failed_shmem3 = 0;

    for (int it = 0; it < iterations; it++) {
        int n = dist_n(rng);
        std::vector<float> input(n);
        for (int i = 0; i < n; i++) input[i] = dist_val(rng);

        float expected;
        {
            auto t0 = Clock::now();
            expected = reduce_reference_tree(input.data(), n);
            auto t1 = Clock::now();
            total_cpu_ms += std::chrono::duration_cast<Ms>(t1 - t0).count();
        }

        float got_naive;
        {
            auto t0 = Clock::now();
            got_naive = reduce(input.data(), n);
            cudaDeviceSynchronize();
            total_naive_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        }

        float got_pingpong;
        {
            auto t0 = Clock::now();
            got_pingpong = reduce_pingpong(input.data(), n);
            cudaDeviceSynchronize();
            total_pingpong_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        }

        float got_shmem;
        {
            auto t0 = Clock::now();
            got_shmem = reduce_in_device<reduce_kernel_shmem>(input.data(), n);
            cudaDeviceSynchronize();
            total_shmem_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        }

        float got_shmem2;
        {
            auto t0 = Clock::now();
            got_shmem2 = reduce_in_device<reduce_kernel_shmem_2>(input.data(), n);
            cudaDeviceSynchronize();
            total_shmem2_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        }
        float got_shmem3;
        {
            auto t0 = Clock::now();
            got_shmem3 = reduce_in_device<reduce_kernel_shmem_3>(input.data(), n);
            cudaDeviceSynchronize();
            total_shmem3_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        }

        auto abs_err = [](float got, float ref) { return std::fabs(got - ref); };
        auto rel_err = [](float got, float ref) {
            return std::fabs(got - ref) / (std::fabs(ref) + 1e-6f);
        };

        if (abs_err(got_naive, expected) > 1e-5f) {
            if (failed_naive < 5)
                std::cerr << "[fuzz/naive] n=" << n << " expected=" << expected << " got=" << got_naive << "\n";
            failed_naive++;
        }
        if (abs_err(got_pingpong, expected) > 1e-5f) {
            if (failed_pingpong < 5)
                std::cerr << "[fuzz/pingpong] n=" << n << " expected=" << expected << " got=" << got_pingpong << "\n";
            failed_pingpong++;
        }
        if (rel_err(got_shmem, expected) > 1e-3f) {
            if (failed_shmem < 5)
                std::cerr << "[fuzz/indevice(shmem)] n=" << n << " expected=" << expected
                          << " got=" << got_shmem << " rel_err=" << rel_err(got_shmem, expected) << "\n";
            failed_shmem++;
        }
        if (rel_err(got_shmem2, expected) > 1e-3f) {
            if (failed_shmem2 < 5)
                std::cerr << "[fuzz/indevice(shmem2)] n=" << n << " expected=" << expected
                          << " got=" << got_shmem2 << " rel_err=" << rel_err(got_shmem2, expected) << "\n";
            failed_shmem2++;
        }
        if (rel_err(got_shmem3, expected) > 1e-3f) {
            if (failed_shmem3 < 5)
                std::cerr << "[fuzz/indevice(shmem3)] n=" << n << " expected=" << expected
                          << " got=" << got_shmem3 << " rel_err=" << rel_err(got_shmem3, expected) << "\n";
            failed_shmem3++;
        }
    }

    std::cout << "[fuzz] cpu-ref  : " << total_cpu_ms     << " ms total, avg " << (total_cpu_ms     / iterations) << " ms\n";
    std::cout << "[fuzz] naive     : " << total_naive_ms   << " ms total, avg " << (total_naive_ms   / iterations) << " ms\n";
    std::cout << "[fuzz] ping-pong : " << total_pingpong_ms<< " ms total, avg " << (total_pingpong_ms/ iterations) << " ms\n";
    std::cout << "[fuzz] indevice(shmem) : " << total_shmem_ms  << " ms total, avg " << (total_shmem_ms / iterations) << " ms\n";
    std::cout << "[fuzz] indevice(shmem2): " << total_shmem2_ms << " ms total, avg " << (total_shmem2_ms/ iterations) << " ms\n";
    std::cout << "[fuzz] indevice(shmem3): " << total_shmem3_ms << " ms total, avg " << (total_shmem3_ms/ iterations) << " ms\n";
    std::cout << "[fuzz] speedup vs cpu-ref: naive " << (total_cpu_ms/total_naive_ms) << "x, ping-pong " << (total_cpu_ms/total_pingpong_ms) << "x, shmem " << (total_cpu_ms/total_shmem_ms) << "x, shmem2 " << (total_cpu_ms/total_shmem2_ms) << "x, shmem3 " << (total_cpu_ms/total_shmem3_ms) << "x\n";
    std::cout << "[fuzz] speedup vs shmem:   naive " << (total_shmem_ms/total_naive_ms) << "x, ping-pong " << (total_shmem_ms/total_pingpong_ms) << "x, shmem2 " << (total_shmem_ms/total_shmem2_ms) << "x, shmem3 " << (total_shmem_ms/total_shmem3_ms) << "x\n";

    if (failed_naive > 0)   std::cerr << "[fuzz] naive failed " << failed_naive << "/" << iterations << "\n";
    if (failed_pingpong > 0) std::cerr << "[fuzz] ping-pong failed " << failed_pingpong << "/" << iterations << "\n";
    if (failed_shmem > 0)   std::cerr << "[fuzz] indevice(shmem) failed " << failed_shmem << "/" << iterations << "\n";
    if (failed_shmem2 > 0)  std::cerr << "[fuzz] indevice(shmem2) failed " << failed_shmem2 << "/" << iterations << "\n";
    if (failed_shmem3 > 0)  std::cerr << "[fuzz] indevice(shmem3) failed " << failed_shmem3 << "/" << iterations << "\n";
    if (failed_naive > 0 || failed_pingpong > 0 || failed_shmem > 0 || failed_shmem2 > 0) return false;

    std::cout << "[fuzz] " << iterations << " iterations passed (max_n=" << max_n
              << ", seed=" << seed << ")\n";
    return true;
}

int main(int argc, char** argv) {
    if (argc > 1 && strcmp(argv[1], "--fuzz") == 0) {
        int iterations = 100;
        int max_n = 1024*1024*2;
        unsigned seed = static_cast<unsigned>(std::random_device{}());
        if (argc > 2) iterations = std::atoi(argv[2]);
        if (argc > 3) max_n = std::atoi(argv[3]);
        if (argc > 4) seed = static_cast<unsigned>(std::atoi(argv[4]));
        return run_fuzz(iterations, max_n, seed) ? 0 : 1;
    }

    int n = 1 << 20;  // 默认 ~100 万元素
    std::cout << "Enter the length of the input array (default " << n << "): ";
    std::cin >> n;

    float* input_h = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) input_h[i] = 0.1f;

    const int REPEATS = 20;
    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    double cpu_ms = 0., naive_ms = 0., pingpong_ms = 0., shmem_ms = 0., shmem2_ms = 0.;
    double shmem3_ms = 0.;
    float r_cpu = 0., r_naive = 0., r_pingpong = 0., r_shmem = 0., r_shmem2 = 0., r_shmem3 = 0.;

    for (int i = 0; i < REPEATS; i++) {
        auto t0 = Clock::now();
        r_cpu = reduce_reference_tree(input_h, n);
        cpu_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    }
    for (int i = 0; i < REPEATS; i++) {
        auto t0 = Clock::now();
        r_naive = reduce(input_h, n);
        cudaDeviceSynchronize();
        naive_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    }
    for (int i = 0; i < REPEATS; i++) {
        auto t0 = Clock::now();
        r_pingpong = reduce_pingpong(input_h, n);
        cudaDeviceSynchronize();
        pingpong_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    }
    for (int i = 0; i < REPEATS; i++) {
        auto t0 = Clock::now();
        r_shmem = reduce_in_device<reduce_kernel_shmem>(input_h, n);
        cudaDeviceSynchronize();
        shmem_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    }
    for (int i = 0; i < REPEATS; i++) {
        auto t0 = Clock::now();
        r_shmem2 = reduce_in_device<reduce_kernel_shmem_2>(input_h, n);
        cudaDeviceSynchronize();
        shmem2_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    }
    for (int i = 0; i < REPEATS; i++) {
        auto t0 = Clock::now();
        r_shmem3 = reduce_in_device<reduce_kernel_shmem_3>(input_h, n);
        cudaDeviceSynchronize();
        shmem3_ms += std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
    }

    std::cout << "n = " << n << ", repeats = " << REPEATS << " (all vs CPU ref)\n";
    std::cout << "cpu-ref   result=" << r_cpu     << "  avg=" << (cpu_ms     / REPEATS) << " ms\n";
    std::cout << "naive     result=" << r_naive  << "  avg=" << (naive_ms   / REPEATS) << " ms\n";
    std::cout << "ping-pong result=" << r_pingpong << "  avg=" << (pingpong_ms / REPEATS) << " ms\n";
    std::cout << "indevice(shmem)  result=" << r_shmem  << "  avg=" << (shmem_ms  / REPEATS) << " ms\n";
    std::cout << "indevice(shmem2) result=" << r_shmem2 << "  avg=" << (shmem2_ms / REPEATS) << " ms\n";
    std::cout << "indevice(shmem3) result=" << r_shmem3 << "  avg=" << (shmem3_ms / REPEATS) << " ms\n";
    std::cout << "speedup vs cpu-ref: naive " << (cpu_ms/naive_ms) << "x, ping-pong " << (cpu_ms/pingpong_ms) << "x, shmem " << (cpu_ms/shmem_ms) << "x, shmem2 " << (cpu_ms/shmem2_ms) << "x\n";
    std::cout << "speedup vs shmem:   naive " << (shmem_ms/naive_ms) << "x, ping-pong " << (shmem_ms/pingpong_ms) << "x, shmem2 " << (shmem_ms/shmem2_ms) << "x\n";
    std::cout << "speedup vs shmem3:   naive " << (shmem3_ms/naive_ms) << "x, ping-pong " << (shmem3_ms/pingpong_ms) << "x, shmem2 " << (shmem3_ms/shmem2_ms) << "x\n";

    free(input_h);
}
