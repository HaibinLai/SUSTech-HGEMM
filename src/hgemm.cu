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
#include "common.h"  // Assuming common.h contains necessary includes and definitions

bool read_matrices_from_dir
(   const std::string& dir,
    std::vector<__half>& A_fp16,
    std::vector<__half>& B_fp16,
    int& M, int& N, int& K
) 
{
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


/////////////////////////////////////////////////////////////////////
//
// Simple CUDA kernel implementations for FP16 matrix multiplication
//
/////////////////////////////////////////////////////////////////////
__global__ void gemm_base_kernel_fp16
(
    const __half* A, 
    const __half* B, 
    __half* C, 
    int M, int N, int K
) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M方向
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N方向

    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; ++k) {
            val += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(val);
    }
}


/////////////////////////////////////////////////////////////////////
//
// Simple CUDA kernel with loop unrolling 
//
/////////////////////////////////////////////////////////////////////
__global__ void gemm_kernel_fp16_unroll
(
    const __half* A, 
    const __half* B, 
    __half* C, 
    int M, int N, int K
) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        #pragma unroll 20
        for (int k = 0; k < K; ++k) {
            val += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(val);
    }
}


#define BLOCK_SIZE 16  // 每个 thread block 计算 C 的 16x16 子块

/////////////////////////////////////////////////////////////////////
//
// Simple CUDA kernel with loop + coalescing
//
/////////////////////////////////////////////////////////////////////
#define COA_BLOCK_SIZE 32  // 每个 thread block 计算 C 的 32x32 子块

__global__ void gemm_kernel_fp16_coalescing
(
    const __half* A,
    const __half* B, 
    __half* C,
    int M, int N, int K
)
{
    int row = blockIdx.x * COA_BLOCK_SIZE + threadIdx.x / COA_BLOCK_SIZE;
    int col = blockIdx.y * COA_BLOCK_SIZE + threadIdx.x % COA_BLOCK_SIZE;

    float sum = 0.0f;

    if (row < M && col < N) {
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            // A[row, k], B[k, col]
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}


/////////////////////////////////////////////////////////////////////
//
// Simple CUDA kernel with loop unrolling + tiled shared memory
//
/////////////////////////////////////////////////////////////////////
__global__ void gemm_kernel_fp16_unroll_tilled
(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* C, 
    int M, int N, int K
) 
{
    // 每一个 block 内的线程id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 每一个 block 要算的 C 的起始位置
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    // 每个线程计算一个 C[row, col] 的值
    float val = 0.0f;

    // 分配 shared memory
    __shared__ __half A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __half B_tile[BLOCK_SIZE][BLOCK_SIZE];

  // 分块遍历 K 维度
    // #pragma unroll
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // A 的 tile 中要加载的列
        int a_col = t * BLOCK_SIZE + tx;
        // B 的 tile 中要加载的行
        int b_row = t * BLOCK_SIZE + ty;

        // 加载 A_tile
        if (row < M && a_col < K) {
            A_tile[ty][tx] = A[row * K + a_col];
        } else {
            A_tile[ty][tx] = __float2half(0.0f);
        }

        // 加载 B_tile
        if (b_row < K && col < N) {
            B_tile[ty][tx] = B[b_row * N + col];
        } else {
            B_tile[ty][tx] = __float2half(0.0f);
        }

        __syncthreads(); // 等待所有线程加载完 tile

        // tile 内计算：遍历 tile 中的 k 维
        // #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            val += __half2float(A_tile[ty][k]) * __half2float(B_tile[k][tx]);
        }

        __syncthreads(); // 等待所有线程计算完当前 tile，才开始加载下一 tile
    }

    // 写回结果到 C
    if (row < M && col < N) {
        C[row * N + col] = __float2half(val);
    }

}


/////////////////////////////////////////////////////////////////////
//
// CUDA kernel with loop unrolling + tiled shared memory + 2 columns
//
/////////////////////////////////////////////////////////////////////
__global__ void gemm_kernel_fp16_tilled_share2cols
(
    const __half* A, 
    const __half* B, 
    __half* C,
    int M, int N, int K
) 
{
    int tx = threadIdx.x;  // 0 ~ BLOCK_SIZE-1
    int ty = threadIdx.y;  // 0 ~ BLOCK_SIZE-1

    // 每个 block 负责 C 矩阵的 tile：高度 BLOCK_SIZE，宽度 BLOCK_SIZE*2
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col0 = (blockIdx.x * BLOCK_SIZE * 2) + tx;        // 当前线程加载的第一列
    int col1 = col0 + BLOCK_SIZE;                         // 当前线程加载的第二列

    // 每个线程负责计算输出 C[row, col0] 和 C[row, col1] 两个元素
    float val0 = 0.0f;
    float val1 = 0.0f;

    // shared memory tile
    __shared__ __half A_tile[BLOCK_SIZE][BLOCK_SIZE];          // tile 大小 M×K 层面的
    __shared__ __half B_tile[BLOCK_SIZE][BLOCK_SIZE*2];       // tile 宽度是 BLOCK_SIZE*2

    #pragma unroll
    for (int t = 0; t < (K+BLOCK_SIZE-1)/BLOCK_SIZE; ++t) {
        int a_col = t * BLOCK_SIZE + tx;

        // 加载 A_tile
        if (row < M && a_col < K) {
            A_tile[ty][tx] = A[row*K + a_col];
        } else {
            A_tile[ty][tx] = __float2half(0.0f);
        }

        // 加载 B_tile：每个线程一次加载两列
        int b_row = t*BLOCK_SIZE + ty;
        if (b_row < K && col0 < N) {
            B_tile[ty][tx] = B[b_row*N + col0];
        } else {
            B_tile[ty][tx] = __float2half(0.0f);
        }
        if (b_row < K && col1 < N) {
            B_tile[ty][tx+BLOCK_SIZE] = B[b_row*N + col1];
        } else {
            B_tile[ty][tx+BLOCK_SIZE] = __float2half(0.0f);
        }

        __syncthreads();

        // 计算 tile 内乘加
        #pragma unroll
        for (int k=0; k<BLOCK_SIZE; ++k) {
            float a_val = __half2float(A_tile[ty][k]);
            val0 += a_val * __half2float(B_tile[k][tx]);
            val1 += a_val * __half2float(B_tile[k][tx+BLOCK_SIZE]);
        }

        __syncthreads();  // 加载下一个 tile 前等待
    }

    // 写回结果到 C
    if (row < M && col0 < N) {
        C[row*N + col0] = __float2half(val0);
    }
    if (row < M && col1 < N) {
        C[row*N + col1] = __float2half(val1);
    }
}




using namespace nvcuda;

// TILE 尺寸：WMMA 固定 tile 是 16x16x16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void hgemm_tensorcore_wmma(
    const __half *A, const __half *B, float *C,
    int M, int N, int K)
{
    // 计算 warp 在 grid 中的位置
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;

    // WMMA fragment：输入和输出
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // 遍历 K 方向的 tiles
    for (int k = 0; k < K; k += WMMA_K) {
        // 计算要加载的地址
        const __half *tile_ptr_A = A + warpM * WMMA_M * K + k;
        const __half *tile_ptr_B = B + k * N + warpN * WMMA_N;

        // Load A row-major, B col-major
        wmma::load_matrix_sync(a_frag, tile_ptr_A, K);
        wmma::load_matrix_sync(b_frag, tile_ptr_B, N);

        // 计算
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 存回 C：float accumulator, row-major
    if (warpM * WMMA_M < M && warpN * WMMA_N < N) {
        float *c_ptr = C + warpM * WMMA_M * N + warpN * WMMA_N;

        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    }
}






// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 00:53:54 on Mon, Feb 13, 2023
//
// Description: wmma async hgemm



/////////////////////////////////////////////////////////////////////
//
// cuBLAS cublasGemmEx FP16 kernel 
//
/////////////////////////////////////////////////////////////////////
void hgemm_kernel_fp16_cublas
(
    const __half* A_fp16, 
    const __half* B_fp16, 
    __half* C_fp16,
    int M, int N, int K
) 
{
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


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
void gemm_coalescing(
    const __half* A_fp16, 
    const __half* B_fp16, 
    __half* C_fp16,
    int M, int N, int K
) 
{
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(CEIL_DIV(N, BLOCK_SIZE), CEIL_DIV(M, BLOCK_SIZE));

    gemm_kernel_fp16_coalescing<<<grid, block>>>(A_fp16, B_fp16, C_fp16, M, N, K);
}





void sustech_hgemm_fp16
(
    const __half* A_fp16, 
    const __half* B_fp16, 
    __half* C_fp16,
    int M, int N, int K
) 
{

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    // dim3 grid((N + 15) / 16, (M + 15) / 16);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // dim3 block(32, 32); // 每个 block 有 1024 线程
    // dim3 grid((N+31)/32, (M+31)/32);

    // hgemm_tensorcore_wmma<<<grid, block>>>(A_fp16, B_fp16, C_fp16, M, N, K);
    // wmma_fp16_gemm_kernel<<<grid, block>>>(A_fp16, B_fp16, C_fp16, M, N, K);

    // gemm_kernel_fp16_unroll<<<grid, block>>>(A_fp16, B_fp16, C_fp16, M, N, K);


    // 选择合适的 kernel 实现
    // if(M >= 20000 && N >= 20000 && K >= 20000){
        // hgemm_kernel_fp16_cublas(A_fp16, B_fp16, C_fp16, M, N, K);
    // }else{

    // gemm_kernel_fp16_tilled_share2cols<<<grid, block>>>(A_fp16, B_fp16, C_fp16, M, N, K);
    // }

    // gemm_coalescing(A_fp16, B_fp16, C_fp16, M, N, K);

    // gemm_kernel_fp16_tilled_share2cols<<<grid, block>>>(A_fp16, B_fp16, C_fp16, M, N, K);
    // gemm_kernel_fp16_unroll_tilled<<<grid, block>>>(A_fp16, B_fp16, C_fp16, M, N, K);

    // gridDim stays the same
    // dim3 gridDim(CEIL_DIV(M, COA_BLOCK_SIZE), CEIL_DIV(N, COA_BLOCK_SIZE));
    // make blockDim 1-dimensional, but don't change number of threads
    // dim3 blockDim(COA_BLOCK_SIZE * COA_BLOCK_SIZE);



}



int main(int argc, char* argv[]) 
{
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


    auto start2 = std::chrono::high_resolution_clock::now();

    sustech_hgemm_fp16(d_A_fp16, d_B_fp16, d_C_custom, M, N, K);

    cudaDeviceSynchronize();
    auto end2 = std::chrono::high_resolution_clock::now();

    double duration_custom = std::chrono::duration<double, std::milli>(end2 - start2).count();

    // 拷贝结果回host计算sum
    // std::vector<__half> C_cublas_host(M * N);
    std::vector<__half> C_custom_host(M * N);
    cudaMemcpy(C_custom_host.data(), d_C_custom, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

    float sum_custom = 0.f;
    for (int i = 0; i < M * N; ++i) {
        // sum_cublas += __half2float(C_cublas_host[i]);
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
