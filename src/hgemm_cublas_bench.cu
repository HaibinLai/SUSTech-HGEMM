#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <chrono>
#include <cuda_fp16.h>  // 用于 __half

// nvcc -O3 -o hgemm_cublas src/hgemm_cublas.cu -lcublas

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

    // 读取 A 
    fa.read(reinterpret_cast<char*>(&m_a), sizeof(int));
    fa.read(reinterpret_cast<char*>(&k_a), sizeof(int));
    size_t size_A = static_cast<size_t>(m_a) * k_a;
    A_fp16.resize(size_A);
    fa.read(reinterpret_cast<char*>(A_fp16.data()), size_A * sizeof(__half));

    // 读取 B 
    fb.read(reinterpret_cast<char*>(&k_b), sizeof(int));
    fb.read(reinterpret_cast<char*>(&n_b), sizeof(int));
    size_t size_B = static_cast<size_t>(k_b) * n_b;
    B_fp16.resize(size_B);
    fb.read(reinterpret_cast<char*>(B_fp16.data()), size_B * sizeof(__half));

    fa.close();
    fb.close();

    // 维度匹配
    if (k_a != k_b) {
        std::cerr << "Error: K dimension mismatch between A and B\n";
        return false;
    }

    M = m_a;
    K = k_a;
    N = n_b;
    return true;
}


void hgemm_cublas_fp16(const __half* A_fp16, const __half* B_fp16, __half* C_fp16,
                       int M, int N, int K) {
    // handle 句柄是一个不透明的指针，用于管理 cuBLAS 库的内部状态和 GPU 资源
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 注意 cuBLAS 是列主序，需要交换 M、N 参数，且输入 A、B 顺序也要对应
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 B_fp16, CUDA_R_16F, N,
                 A_fp16, CUDA_R_16F, K,
                 &beta,
                 C_fp16, CUDA_R_16F, N,
                 CUDA_R_32F,  // 计算时用 FP32 计算精度
                 CUBLAS_GEMM_DFALT_TENSOR_OP);  // Tensor Core 路径

    cublasDestroy(handle);
}


int main(int argc, char* argv[]) {
    std::string input_dir = "data/input/Case1_768x768x768";
    std::string output_dir = "data/output";

    // 解析命令行参数
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

    // 从目录名中提取 case 名字
    std::string case_name = input_dir.substr(input_dir.find_last_of("/\\") + 1);
    std::string output_file = output_dir + "/result_" + case_name + ".txt";

    int M, N, K;
    std::vector<__half> A_fp16, B_fp16;

    std::cout << "===== Processing case: " << case_name << " =====" << std::endl;
    std::cout << "Reading Input directory: " << input_dir << " ..." << std::endl;

    if (!read_matrices_from_dir(input_dir, A_fp16, B_fp16, M, N, K)) {
        return 1;
    }

    __half *d_A_fp16 = nullptr, *d_B_fp16 = nullptr, *d_C_fp16 = nullptr;
    cudaMalloc(&d_A_fp16, M * K * sizeof(__half));
    cudaMalloc(&d_B_fp16, K * N * sizeof(__half));
    cudaMalloc(&d_C_fp16, M * N * sizeof(__half));

    cudaMemcpy(d_A_fp16, A_fp16.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_fp16, B_fp16.data(), K * N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C_fp16, 0, M * N * sizeof(__half));

    cudaDeviceSynchronize();

    std::cout << "Read data complete, "
              << "M: " << M << ", N: " << N << ", K: " << K << std::endl;

    double average_time = 0.0;
    for(int i = 0; i<10; i++){
        auto start = std::chrono::high_resolution_clock::now();
        hgemm_cublas_fp16(d_A_fp16, d_B_fp16, d_C_fp16, M, N, K);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        average_time += duration;
        // std::cout << "Iteration " << i + 1 << ": " << duration << " ms" << std::endl;
    }    

    average_time /= 10.0f;

    double flops = 2.0 * M * N * K;
    double gflops = flops / (average_time / 1000.0) / 1e9;

    // 拷贝结果回 host
    std::vector<__half> C_fp16_host(M * N);
    cudaMemcpy(C_fp16_host.data(), d_C_fp16, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < M * N; ++i)
        sum += __half2float(C_fp16_host[i]);

    std::cout << "cuBLAS FP16 Avg GEMM Time: " << average_time << " ms, "
              << "Avg gFLOPS: " << gflops << std::endl;
    std::cout << "Result sum: " << sum << std::endl;

    int M_limit = std::min(M, 10);
    int N_limit = std::min(N, 10);

    std::ofstream outfile(output_file);
    if (outfile.is_open()) {
        outfile << "Performance Metrics:\n";
        outfile << "Time (ms): " << average_time << "\n";
        outfile << "GFLOPS: " << gflops << "\n";
        outfile << "M: " << M << ", N: " << N << ", K: " << K << "\n";
        outfile << "Result Matrix (C) (partial):\n";
        for (int i = 0; i < M_limit; ++i) {
            for (int j = 0; j < N_limit; ++j) {
                outfile << __half2float(C_fp16_host[i * N + j]) << " ";
            }
            outfile << "\n";
        }
        outfile.close();
    } else {
        std::cerr << "Error: Cannot open " << output_file << " for writing" << std::endl;
    }

    cudaFree(d_A_fp16);
    cudaFree(d_B_fp16);
    cudaFree(d_C_fp16);

    std::cout << "===== Finished processing case: " << case_name << " =====" << std::endl;

    return 0;
}