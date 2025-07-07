#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>
#include <chrono>
#include <cuda_fp16.h>  // 用于 __half

// nvcc -O3 -o hgemm_cublas_load src/hgemm_read.cu -lcublas

// 从文件读取矩阵
bool read_matrices_from_file(const std::string& filename,
                             std::vector<__half>& A_fp16,
                             std::vector<__half>& B_fp16,
                             int& M, int& N, int& K) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    if (!(iss >> M >> N >> K)) {
        std::cerr << "Error: Invalid header line" << std::endl;
        return false;
    }

    std::cout << "Matrix dimensions: M = " << M << ", N = " << N << ", K = " << K << std::endl;

    A_fp16.resize(M * K);
    B_fp16.resize(K * N);

    for (int i = 0; i < M * K; ++i) {
        float value;
        if (!(file >> value)) {
            std::cerr << "Error at A[" << i << "], expected total: " << M * K << std::endl;
            return false;
        }
        A_fp16[i] = __float2half(value);
    }


    for (int i = 0; i < K * N; ++i) {
        float val;
        if (!(file >> val)) {
            std::cerr << "Error: Not enough B values\n";
            return false;
        }
        B_fp16[i] = static_cast<__half>(val);
    }

    return true;
}

void hgemm_cublas(const float* A, const float* B, float* C,
                  int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N, // 注意 col-major vs row-major
                N, M, K, 
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);

    cublasDestroy(handle);
}


int main(int argc, char* argv[]) {
    std::string input_file = "data/input/matrices_Case1_768x768x768.txt";
    std::string output_file = "data/output/result_Case1.txt";

    // parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_file = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [-i input_file] [-o output_file]" << std::endl;
            return 1;
        }
    }

    int M, N, K;
    std::vector<__half> A_fp16, B_fp16;

    if (!read_matrices_from_file(input_file, A_fp16, B_fp16, M, N, K)) {
        return 1;
    }

    float *A_fp32, *B_fp32, *C;
    cudaMallocManaged(&A_fp32, M * K * sizeof(float));
    cudaMallocManaged(&B_fp32, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    for (int i = 0; i < M * K; ++i)
        A_fp32[i] = static_cast<float>(A_fp16[i]);
    for (int i = 0; i < K * N; ++i)
        B_fp32[i] = static_cast<float>(B_fp16[i]);
    for (int i = 0; i < M * N; ++i)
        C[i] = 0.0f;

    cudaDeviceSynchronize();

    auto start = std::chrono::high_resolution_clock::now();
    hgemm_cublas(A_fp32, B_fp32, C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    double flops = 2.0 * M * N * K;
    double gflops = flops / (duration / 1000.0) / 1e9;

    std::cout << "CuBLAS GEMM Time: " << duration << " ms, "
              << "gFLOPS: " << gflops << std::endl;

    float sum = 0.0f;
    for (int i = 0; i < M * N; ++i)
        sum += C[i];
    std::cout << "Result sum: " << sum << std::endl;

    std::ofstream outfile(output_file);
    if (outfile.is_open()) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                outfile << C[i * N + j] << " ";
            }
            outfile << "\n";
        }
        outfile.close();
    } else {
        std::cerr << "Error: Cannot open " << output_file << " for writing" << std::endl;
    }

    cudaFree(A_fp32);
    cudaFree(B_fp32);
    cudaFree(C);

    return 0;
}