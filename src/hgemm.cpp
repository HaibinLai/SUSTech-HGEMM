#include <immintrin.h> // AVX2 and F16C
#include <omp.h>
#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <fstream>
#include <sstream>

// 定义FP16类型
using float16_t = _Float16;

// AVX2内核：计算C[m:m+8, n:n+8] += A[m:m+8, k:k+K] × B[k:k+K, n:n+8]
void hgemm_kernel_avx2(const float16_t* A, const float16_t* B, float* C,
                       int M, int N, int K, int lda, int ldb, int ldc,
                       int m_start, int m_end, int n_start, int n_end) {
    for (int m = m_start; m < m_end; m += 8) { // 每次处理8行
        for (int n = n_start; n < n_end; n += 8) { // 每次处理8列
            // 累加器：FP32，8x8子块
            __m256 acc[8][8];
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j)
                    acc[i][j] = _mm256_setzero_ps();

            // 沿K维度计算
            for (int k = 0; k < K; ++k) {
                // 加载A的8个FP16值（一行）
                __m128i a_ph[8];
                for (int i = 0; i < 8 && m + i < M; ++i) {
                    a_ph[i] = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&A[(m + i) * lda + k]));
                }

                // 加载B的8个FP16值（一列）
                __m128i b_ph[8];
                for (int j = 0; j < 8 && n + j < N; ++j) {
                    b_ph[j] = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&B[k * ldb + n + j]));
                }

                // 转换为FP32并计算
                for (int i = 0; i < 8 && m + i < M; ++i) {
                    __m256 a_ps = _mm256_cvtph_ps(a_ph[i]); // FP16 -> FP32
                    for (int j = 0; j < 8 && n + j < N; ++j) {
                        __m256 b_ps = _mm256_cvtph_ps(b_ph[j]);
                        acc[i][j] = _mm256_fmadd_ps(a_ps, b_ps, acc[i][j]);
                    }
                }
            }

            // 存储结果到C
            for (int i = 0; i < 8 && m + i < M; ++i) {
                for (int j = 0; j < 8 && n + j < N; ++j) {
                    float* acc_ptr = reinterpret_cast<float*>(&acc[i][j]);
                    for (int l = 0; l < 8; ++l) {
                        C[(m + i) * ldc + (n + j)] += acc_ptr[l];
                    }
                }
            }
        }
    }
}

// 并行HGEMM
void hgemm_parallel(const float16_t* A, const float16_t* B, float* C,
                    int M, int N, int K, int lda, int ldb, int ldc) {
    const int BLOCK_SIZE = 128; // 缓存友好的分块大小
#pragma omp parallel for collapse(2) num_threads(24)
    for (int m = 0; m < M; m += BLOCK_SIZE) {
        for (int n = 0; n < N; n += BLOCK_SIZE) {
            int m_end = std::min(m + BLOCK_SIZE, M);
            int n_end = std::min(n + BLOCK_SIZE, N);
            hgemm_kernel_avx2(A, B, C, M, N, K, lda, ldb, ldc,
                              m, m_end, n, n_end);
        }
    }
}

// 从文件读取矩阵
bool read_matrices_from_file(const std::string& filename,
                           std::vector<float16_t>& A,
                           std::vector<float16_t>& B,
                           int& M, int& N, int& K) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    // 读取M, N, K
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    if (!(iss >> M >> N >> K)) {
        std::cerr << "Error: Invalid format for M, N, K" << std::endl;
        return false;
    }

    // 分配矩阵空间
    A.resize(M * K);
    B.resize(K * N);

    // 读取矩阵A
    for (int i = 0; i < M * K; ++i) {
        float value;
        if (!(file >> value)) {
            std::cerr << "Error: Not enough values for matrix A" << std::endl;
            return false;
        }
        A[i] = static_cast<float16_t>(value);
    }

    // 读取矩阵B
    for (int i = 0; i < K * N; ++i) {
        float value;
        if (!(file >> value)) {
            std::cerr << "Error: Not enough values for matrix B" << std::endl;
            return false;
        }
        B[i] = static_cast<float16_t>(value);
    }

    file.close();
    return true;
}

// 测试代码
int main() {
    int M, N, K;
    std::vector<float16_t> A, B;
    std::string filename = "matrices.txt";

    // 从文件读取矩阵
    if (!read_matrices_from_file(filename, A, B, M, N, K)) {
        return 1;
    }

    std::vector<float> C(M * N, 0.0f);

    // 运行HGEMM
    auto start = std::chrono::high_resolution_clock::now();
    hgemm_parallel(A.data(), B.data(), C.data(), M, N, K, K, N, N);
    auto end = std::chrono::high_resolution_clock::now();

    // 计算时间和TFLOPS
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    double flops = 2.0 * M * N * K;
    double gflops = (flops / (duration / 1000.0)) / 1e9;
    std::cout << "HGEMM Time: " << duration << " ms, "
              << "gFLOPS: " << gflops << std::endl;

    // 验证结果（求和检查）
    float sum = 0.0f;
    for (float x : C) sum += x;
    std::cout << "Result sum: " << sum << std::endl;

    return 0;
}