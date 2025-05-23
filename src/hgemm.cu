#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <chrono>

#define CHECK_CUDA(call) \
do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 基础矩阵乘法核函数（未优化）
__global__ void hgemm_baseline(
    const half *A,  // M x K (row-major)
    const half *B,  // K x N (row-major)
    float *C,       // M x N (row-major)
    int M, int N, int K) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            half a = A[row * K + k];
            half b = B[k * N + col];
            sum += __half2float(a) * __half2float(b);
        }
        C[row * N + col] = sum;
    }
}

// 生成随机FP16矩阵并保存到文件
void generate_matrix(const char* filename, int rows, int cols) {
    std::ofstream out(filename, std::ios::binary);
    half *h_data = new half[rows * cols];
    
    for (int i = 0; i < rows * cols; ++i) {
        h_data[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
    }
    
    out.write(reinterpret_cast<char*>(h_data), rows * cols * sizeof(half));
    delete[] h_data;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " M N K input_dir" << std::endl;
        return EXIT_FAILURE;
    }

    const int M = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int K = atoi(argv[3]);
    const char* input_dir = argv[4];

    // 生成测试数据（实际比赛应从文件读取）
    generate_matrix((std::string(input_dir) + "/A.bin").c_str(), M, K);
    generate_matrix((std::string(input_dir) + "/B.bin").c_str(), K, N);

    // 加载输入矩阵
    half *h_A = new half[M*K];
    half *h_B = new half[K*N];
    float *h_C = new float[M*N];
    
    std::ifstream finA(std::string(input_dir) + "/A.bin", std::ios::binary);
    std::ifstream finB(std::string(input_dir) + "/B.bin", std::ios::binary);
    finA.read(reinterpret_cast<char*>(h_A), M*K*sizeof(half));
    finB.read(reinterpret_cast<char*>(h_B), K*N*sizeof(half));

    // 分配设备内存
    half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M*K*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_B, K*N*sizeof(half)));
    CHECK_CUDA(cudaMalloc(&d_C, M*N*sizeof(float)));

    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice));

    // 配置核函数参数
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 执行并计时
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    CHECK_CUDA(cudaEventRecord(start));
    hgemm_baseline<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // 计算性能
    double flops = 2.0 * M * N * K;
    double tflops = (flops / 1e12) / (ms / 1e3);
    printf("Performance: %.2f ms, %.2f TFLOPS\n", ms, tflops);

    // 写回结果
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));
    // 生成文件名，格式如 result_512_768_3072.bin
    std::string filename = "result_" + std::to_string(M) + "_" + 
                          std::to_string(N) + "_" + std::to_string(K) + ".bin";

    std::ofstream fout(filename, std::ios::binary);
    fout.write(reinterpret_cast<char*>(h_C), M*N*sizeof(float));

    // 清理资源
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}