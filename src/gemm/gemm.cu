#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUBLAS_CHECK(call)                                                   \
    do {                                                                     \
        cublasStatus_t _s = (call);                                          \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error: " << _s                              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(_e)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

__global__ void gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
__global__ void gemm_2d_tile(float* A, float* B, float* C, int M, int N, int K) {
    const int TILE_SIZE = 32;
    __shared__ float a_tile[TILE_SIZE * TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE * TILE_SIZE];
     int col = blockIdx.x * blockDim.x + threadIdx.x;
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     C[row * N + col] = 0.0f;

     for(int t = 0; t < K; t += TILE_SIZE) {
        a_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = A[row * K + t + threadIdx.x];
        // A little bit tricky here, but it's correct.
        b_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = B[(t + threadIdx.y) * N + col];
        __syncthreads();
        float sum = 0.0f;
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += a_tile[threadIdx.y * TILE_SIZE + k] * b_tile[k * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
        if(row < M && col < N) {
            C[row * N + col] += sum;
        }
     }
}



template<void (*GEMM_KERNEL)(float*, float*, float*, int, int, int)>
void run_gemm(float* A, float* B, float* C, int M, int N, int K, int bx = 16, int by = 16) {
    dim3 block(bx, by);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&B_d, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&C_d, sizeof(float) * M * N));
    CUDA_CHECK(cudaMemcpy(A_d, A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    GEMM_KERNEL<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(C, C_d, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
}

void gemm_cpu(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

static void fill_random(std::vector<float>& v, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
}

static bool compare_close(const std::vector<float>& got,
                          const std::vector<float>& ref,
                          int M, int N,
                          bool verbose = false,
                          float rtol = 1e-3f,
                          float atol = 1e-3f) {
    if (got.size() != ref.size()) return false;
    float max_abs = 0.f;
    float max_rel = 0.f;
    int max_i = -1;
    for (int i = 0; i < (int)got.size(); i++) {
        float a = got[i];
        float b = ref[i];
        float abs_err = std::fabs(a - b);
        float rel_err = abs_err / (std::fabs(b) + 1e-6f);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            max_rel = rel_err;
            max_i = i;
        }
        if (!(abs_err <= atol || rel_err <= rtol)) {
            int row = i / N;
            int col = i % N;
            std::cerr << "[check] mismatch at (" << row << "," << col << ") "
                      << "ref=" << b << " got=" << a
                      << " abs=" << abs_err << " rel=" << rel_err << "\n";
            if (verbose && max_i >= 0) {
                int max_row = max_i / N;
                int max_col = max_i % N;
                std::cerr << "[check] max_err so far at (" << max_row << "," << max_col << ") "
                          << "abs=" << max_abs << " rel=" << max_rel << "\n";
            }
            return false;
        }
    }
    if (verbose && max_i >= 0) {
        int row = max_i / N;
        int col = max_i % N;
        std::cerr << "[check] max_err at (" << row << "," << col << ") "
                  << "abs=" << max_abs << " rel=" << max_rel << "\n";
    }
    return true;
}

template<void (*GEMM_KERNEL)(float*, float*, float*, int, int, int)>
static float bench_kernel_ms_device(const float* A_h, const float* B_h, float* C_h,
                                    int M, int N, int K,
                                    int warmup, int repeats,
                                    int bx = 16, int by = 16) {
    const size_t bytesA = sizeof(float) * (size_t)M * (size_t)K;
    const size_t bytesB = sizeof(float) * (size_t)K * (size_t)N;
    const size_t bytesC = sizeof(float) * (size_t)M * (size_t)N;

    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, bytesA));
    CUDA_CHECK(cudaMalloc(&B_d, bytesB));
    CUDA_CHECK(cudaMalloc(&C_d, bytesC));
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C_d, 0, bytesC));

    dim3 block(bx, by);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    for (int i = 0; i < warmup; i++) {
        GEMM_KERNEL<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) {
        GEMM_KERNEL<<<grid, block>>>(A_d, B_d, C_d, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return ms / repeats;
}

static float bench_cublas_ms(const float* A_h, const float* B_h, float* C_h,
                             int M, int N, int K, int warmup, int repeats) {
    const size_t bytesA = sizeof(float) * (size_t)M * (size_t)K;
    const size_t bytesB = sizeof(float) * (size_t)K * (size_t)N;
    const size_t bytesC = sizeof(float) * (size_t)M * (size_t)N;

    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    CUDA_CHECK(cudaMalloc(&A_d, bytesA));
    CUDA_CHECK(cudaMalloc(&B_d, bytesB));
    CUDA_CHECK(cudaMalloc(&C_d, bytesC));
    CUDA_CHECK(cudaMemcpy(A_d, A_h, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(C_d, 0, bytesC));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    // 行主序 C=A*B 等价于列主序 C^T = B^T * A^T
    // cublasSgemm: C_col = A_col * B_col，维度为 (N,M) = (N,K)*(K,M)
    auto sgemm = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha, B_d, N,
                    A_d, K,
            &beta,  C_d, N));
    };

    for (int i = 0; i < warmup; i++) sgemm();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repeats; i++) sgemm();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(C_h, C_d, bytesC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return ms / repeats;
}

static void usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " --bench [M=4096] [N=4096] [K=4096] [warmup=5] [repeats=20]\n"
        << "  " << prog << " --check [M=256] [N=256] [K=256]\n"
        << "  " << prog << "            (no args -> same as --bench defaults)\n";
}

int main(int argc, char** argv) {
    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 || std::strcmp(argv[1], "--help") == 0)) {
        usage(argv[0]);
        return 0;
    }

    bool do_check = false;
    int M = 4096, N = 4096, K = 4096;
    int warmup = 5, repeats = 20;
    if (argc > 1 && std::strcmp(argv[1], "--check") == 0) {
        do_check = true;
        M = 256;
        N = 256;
        K = 256;
        if (argc > 2) M = std::atoi(argv[2]);
        if (argc > 3) N = std::atoi(argv[3]);
        if (argc > 4) K = std::atoi(argv[4]);
        warmup = 0;
        repeats = 1;
    } else if (argc > 1 && std::strcmp(argv[1], "--bench") == 0) {
        do_check = false;
        if (argc > 2) M = std::atoi(argv[2]);
        if (argc > 3) N = std::atoi(argv[3]);
        if (argc > 4) K = std::atoi(argv[4]);
        if (argc > 5) warmup = std::atoi(argv[5]);
        if (argc > 6) repeats = std::atoi(argv[6]);
    } else if (argc > 1) {
        // 兼容：直接给 M N K
        do_check = false;
        M = std::atoi(argv[1]);
        if (argc > 2) N = std::atoi(argv[2]);
        if (argc > 3) K = std::atoi(argv[3]);
    }

    std::mt19937 rng(static_cast<unsigned>(std::random_device{}()));
    std::vector<float> A((size_t)M * (size_t)K);
    std::vector<float> B((size_t)K * (size_t)N);
    std::vector<float> C_ref((size_t)M * (size_t)N, 0.f);
    std::vector<float> C_gpu((size_t)M * (size_t)N, 0.f);

    fill_random(A, rng);
    fill_random(B, rng);

    using Clock = std::chrono::high_resolution_clock;
    using Ms = std::chrono::duration<double, std::milli>;

    if (do_check) {
        // 正确性：只做一次（大尺寸请用 --check 单独跑，默认 bench 会跳过）
        auto t0 = Clock::now();
        gemm_cpu(A.data(), B.data(), C_ref.data(), M, N, K);
        double cpu_once_ms = std::chrono::duration_cast<Ms>(Clock::now() - t0).count();
        std::cout << "[check] cpu M=" << M << " N=" << N << " K=" << K
                  << " cpu_once=" << cpu_once_ms << " ms\n";

        auto do_check_kernel = [&](const char* name, std::vector<float>& result,
                                   bool ok) {
            std::cout << "[check] " << name
                      << " M=" << M << " N=" << N << " K=" << K
                      << " => " << (ok ? "PASS" : "FAIL") << "\n";
            return ok;
        };

        std::fill(C_gpu.begin(), C_gpu.end(), 0.f);
        run_gemm<gemm_naive>(A.data(), B.data(), C_gpu.data(), M, N, K, 16, 16);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!do_check_kernel("gemm_naive", C_gpu, compare_close(C_gpu, C_ref, M, N, true))) return 1;

        std::fill(C_gpu.begin(), C_gpu.end(), 0.f);
        run_gemm<gemm_2d_tile>(A.data(), B.data(), C_gpu.data(), M, N, K, 32, 32);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!do_check_kernel("gemm_2d_tile", C_gpu, compare_close(C_gpu, C_ref, M, N, true))) return 1;

        std::fill(C_gpu.begin(), C_gpu.end(), 0.f);
        bench_cublas_ms(A.data(), B.data(), C_gpu.data(), M, N, K, 0, 1);
        if (!do_check_kernel("cublas",       C_gpu, compare_close(C_gpu, C_ref, M, N, true))) return 1;
    }

    // 误差：以 gemm_naive 输出为基准，对比其他实现
    std::vector<float> C_naive((size_t)M * (size_t)N, 0.f);
    std::vector<float> C_tile ((size_t)M * (size_t)N, 0.f);
    std::vector<float> C_cublasv((size_t)M * (size_t)N, 0.f);
    run_gemm<gemm_naive>  (A.data(), B.data(), C_naive.data(),  M, N, K, 16, 16);
    run_gemm<gemm_2d_tile>(A.data(), B.data(), C_tile.data(),   M, N, K, 32, 32);
    bench_cublas_ms       (A.data(), B.data(), C_cublasv.data(), M, N, K, 0, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto print_diff = [&](const char* name, const std::vector<float>& got,
                          const std::vector<float>& ref) {
        float max_abs = 0.f, max_rel = 0.f;
        for (int i = 0; i < M * N; i++) {
            float diff = std::fabs(got[i] - ref[i]);
            float rel  = diff / (std::fabs(ref[i]) + 1e-6f);
            if (diff > max_abs) { max_abs = diff; max_rel = rel; }
        }
        std::cout << "[diff]  " << name << " vs gemm_naive"
                  << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
    };
    print_diff("gemm_2d_tile", C_tile,   C_naive);
    print_diff("cublas",       C_cublasv, C_naive);

    // 性能：kernel-only（显存常驻 + CUDA event）
    double flops = 2.0 * (double)M * (double)N * (double)K;

    float ms_naive  = bench_kernel_ms_device<gemm_naive>(
        A.data(), B.data(), C_gpu.data(), M, N, K, warmup, repeats, 16, 16);
    float ms_tile   = bench_kernel_ms_device<gemm_2d_tile>(
        A.data(), B.data(), C_gpu.data(), M, N, K, warmup, repeats, 32, 32);
    float ms_cublas = bench_cublas_ms(
        A.data(), B.data(), C_gpu.data(), M, N, K, warmup, repeats);

    auto print_bench = [&](const char* name, float avg_ms, float speedup) {
        double gflops = (flops / (avg_ms * 1e-3)) / 1e9;
        std::cout << "[bench] " << name
                  << " M=" << M << " N=" << N << " K=" << K
                  << " avg=" << avg_ms << " ms"
                  << " GFLOPS=" << gflops;
        if (speedup > 0.f) std::cout << " speedup=" << speedup << "x";
        std::cout << "\n";
    };

    print_bench("gemm_naive",   ms_naive,  0.f);
    print_bench("gemm_2d_tile", ms_tile,   ms_naive / ms_tile);
    print_bench("cublas",       ms_cublas, ms_naive / ms_cublas);

    return 0;
}
