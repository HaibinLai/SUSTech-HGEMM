#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <chrono>
#include <cuda_fp16.h>  // for __half

bool read_matrices_from_dir(const std::string& dir,
                            std::vector<__half>& A_fp16,
                            std::vector<__half>& B_fp16,
                            int& M, int& N, int& K) {
    std::string path_A = dir + "/A_matrix.bin";
    std::string path_B = dir + "/B_matrix.bin";

    std::ifstream fa(path_A, std::ios::binary);
    std::ifstream fb(path_B, std::ios::binary);
    if (!fa.is_open() || !fb.is_open()) {
        std::cerr << "Error opening binary matrix files in " << dir << std::endl;
        return false;
    }

    int m_a = 0, k_a = 0, k_b = 0, n_b = 0;

    fa.read(reinterpret_cast<char*>(&m_a), sizeof(int));
    fa.read(reinterpret_cast<char*>(&k_a), sizeof(int));
    size_t size_A = static_cast<size_t>(m_a) * k_a;
    A_fp16.resize(size_A);
    fa.read(reinterpret_cast<char*>(A_fp16.data()), size_A * sizeof(__half));

    fb.read(reinterpret_cast<char*>(&k_b), sizeof(int));
    fb.read(reinterpret_cast<char*>(&n_b), sizeof(int));
    size_t size_B = static_cast<size_t>(k_b) * n_b;
    B_fp16.resize(size_B);
    fb.read(reinterpret_cast<char*>(B_fp16.data()), size_B * sizeof(__half));

    fa.close();
    fb.close();

    if (k_a != k_b) {
        std::cerr << "Error: K dimension mismatch between A and B\n";
        return false;
    }

    M = m_a;
    K = k_a;
    N = n_b;
    return true;
}

__global__ void simple_gemm_kernel_fp16(const __half* A, const __half* B, __half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; ++k) {
            val += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(val);
    }
}



int main(int argc, char* argv[]) {
    std::string input_dir = "data/input/Case1_768x768x768";
    std::string output_dir = "data/output";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-d" || arg == "--indir") && i + 1 < argc) {
            input_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--outdir") && i + 1 < argc) {
            output_dir = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [-d input_dir] [-o output_dir]" << std::endl;
            return 1;
        }
    }

    std::string case_name = input_dir.substr(input_dir.find_last_of("/\\") + 1);
    std::string output_file = output_dir + "/result_" + case_name + ".txt";

    int M, N, K;
    std::vector<__half> A_fp16, B_fp16;
    if (!read_matrices_from_dir(input_dir, A_fp16, B_fp16, M, N, K)) return 1;

    __half *d_A_fp16, *d_B_fp16,  *d_C_custom;
    cudaMalloc(&d_A_fp16, M * K * sizeof(__half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(__half));
    cudaMalloc(&d_C_custom, M * N * sizeof(__half));

    cudaMemcpy(d_A_fp16, A_fp16.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, B_fp16.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C_custom, 0, M * N * sizeof(__half));
    cudaDeviceSynchronize();


    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    auto start2 = std::chrono::high_resolution_clock::now();
    simple_gemm_kernel_fp16<<<grid, block>>>(d_A_fp16, d_B_fp16, d_C_custom, M, N, K);
    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();

    double duration_custom = std::chrono::duration<double, std::milli>(end2 - start2).count();

    // 拷贝结果回host计算sum
    std::vector<__half> C_cublas_host(M * N);
    std::vector<__half> C_custom_host(M * N);
    cudaMemcpy(C_custom_host.data(), d_C_custom, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    float sum_cublas = 0.f, sum_custom = 0.f;
    for (int i = 0; i < M * N; ++i) {
        sum_cublas += __half2float(C_cublas_host[i]);
        sum_custom += __half2float(C_custom_host[i]);
    }

    double flops = 2.0 * M * N * K;
    double custom_gflops = flops / (duration_custom / 1000.0) / 1e9;

    std::cout << "Custom FP16 Kernel Time: " << duration_custom << " ms, gFLOPS: " << custom_gflops << std::endl;
    std::cout << "Custom Kernel Result sum: " << sum_custom << std::endl;

    std::ofstream out(output_file);
    if (out.is_open()) {
        out << "Case: " << case_name << "\n";
        out << "Custom FP16 Kernel Time: " << duration_custom << " ms, gFLOPS: " << custom_gflops << "\n";
        out << "Custom Kernel Result sum: " << sum_custom << "\n";
        out.close();
    }

    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_custom);

    return 0;
}
