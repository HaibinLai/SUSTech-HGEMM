
# 南方科技大学 HPC 校内赛 - GPU-HGEMM 加速赛题

**联系人**：赖海斌 12211612@mail.sustech.edu.cn  
**硬件平台**：NVIDIA V100 GPU (32GB显存) \* 1 + Xeon Platinum CPU

## 一、任务说明

### 1.1 赛题背景

本次赛题要求在一张 NVIDIA V100 GPU上加速半精度通用矩阵乘法（HGEMM，Half-Precision General Matrix Multiplication）。HGEMM是矩阵乘法的一种形式，使用16位浮点数（half-precision, FP16）进行计算，适用于高性能计算场景。

GEMM（General Matrix Multiplication）通用矩阵乘法是科学计算与深度学习领域最核心的运算之一，其标准形式为：

```
C = α * A * B + β * C
```

其中：

- A 是维度为 M×K 的输入矩阵

- B 是维度为 K×N 的输入矩阵

- C 是维度为 M×N 的输入/输出矩阵

- α, β 是浮点数标量系数

在本次赛题中，我们取 α=1 , β=0，即只考虑两矩阵相乘的情况, 即最简单的情况：

```
C = A * B
```


### 1.2 赛题要求

本题需要大家完成两阶段的计算，允许大家使用除了cublas以外的任意的外部库，欢迎大家学习参考网上其他开源代码（如有使用，请在相应代码位置及报告中说明，否则视为抄袭）。

**阶段一（基础部分 80%）：**  
纯GPU计算，要求实现以下特性：
- 使用TensorCore加速
- 支持矩阵尺寸对齐（16的倍数）
- 实现至少2种优化策略（如共享内存、双缓冲等）

**阶段二（挑战部分 20%）：**  
GPU-CPU协同计算：
- 当矩阵所需显存 > 24GB时自动启用
- 实现分块流水线传输（overlap拷贝与计算）
- 支持非对齐尺寸（任意M/N/K）

### 1.3 测试案例
| 案例编号 | M    | N     | K     | 阶段 | 案例说明           | 现实应用|
|----------|-------|-------|-------|------|--------------------|----|
| Case1    | 768   | 768   | 768   | 1    | 基准测试           |Transformer 注意力 |
| Case2    | 512   | 3072   | 1024   | 1    | 基准测试           |神经网络分类头 |
| Case3    | 3136   | 576   | 64   | 1    | 非对称矩阵           | ResNet50 3x3 卷积 |
| Case4    | 4,096  | 4,096  | 4,096  | 1    | 基准测试         | |
| Case5    | 16384   | 128  | 16384  | 1    | 非对称矩阵         | LLama3 8B 注意力（batch size=32） |
| Case6    | 16384   | 4096  | 14336  | 1    | 非对称矩阵         | LLama3 8B FFN（batch size=32） |
| Case7    | 32,768 | 32,768 | 32,768 | 1    | 大矩阵测试     | HPL-MxP Benchmark |
| Case8    | 81920 | 81920 | 81920 | 2    | GPU-CPU协同      |    |
| Case8    | 102400 | 102400 | 102400 | 2    | GPU-CPU协同      |    |


## 二、评分标准
### 2.1 技术报告（50%）
| 评分项                | 比例 | 要求                                                                 |
|-----------------------|------|----------------------------------------------------------------------|
| GPU硬件优化策略       | 30%  | 需说明采用的优化策略如TensorCore使用、pipeline优化等 及其优化原理                 |
| 优化策略代码说明          | 20%  | 需在策略后列出对应关键代码片段                                           |
| Nsight性能分析        | 20%  | 包含SM利用率、显存带宽等指标截图                 |
| 实验分析	| 10%	|需对优化技术进行组合对比（如采用消融实验），展示不同配置下在不同矩阵大小的性能变化规律 |
| 文献引用              | 10%  | 正确引用相关技术文献                                                 |

### 2.2 程序得分（50%）
<!-- 基准公式：
```
Score = 70*(YourPerf / MaxPerf) + 30*(1 - Error)
```
其中：
- MaxPerf：所有参赛队伍最佳性能
- Error：与cuBLAS结果的相对误差 -->
**程序得分**由基础阶段（80%）和进阶阶段（20%）加权构成。每阶段分为30%的正确性得分与70%性能得分。

$$
\text{TotalScore} = 0.8 \times \text{BaseScore} + 0.2 \times \text{AdvancedScore}
$$

### 基础阶段评分（ $$N_1$$ 个测试点）
$$
\text{BaseScore} = \frac{1}{N_1} \sum_{i=1}^{N_1} \left[ 70 \times \left(\frac{\text{Perf}_i}{\text{MaxPerf}_i}\right) + 30 \times \left(1 - \text{RelError}_i\right) \right]
$$

### 进阶阶段评分（ $$N_2$$ 个测试点）
$$
\text{AdvancedScore} = \frac{1}{N_2} \sum_{j=1}^{N_2} \left[ 70 \times \left(\frac{\text{Perf}_j}{\text{MaxPerf}_j}\right) + 30 \times \left(1 - \text{RelError}_j\right) \right]
$$

## 关键指标定义
- **性能比** 
$$\frac{\text{Perf}}{\text{MaxPerf}}$$ 
  - 实际性能（如GFLOPS）与理论峰值性能的比值
- **相对误差 RelError**    
  - 基于Frobenius范数的GEMM结果误差（参考值 vs 计算结果）：

$`\text{RelError} = \frac{\|\mathbf{A}_{\text{ref}} - \mathbf{A}_{\text{calc}}\|_F}{\|\mathbf{A}_{\text{ref}}\|_F}`$

  - **误差容忍**：$`\text{RelError} > 0.05`$时，该测试点得分为0




## 三、技术路线建议
### 3.1 核心优化策略
```cpp
// 示例：双缓冲+异步拷贝核心结构
template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
__global__ void hgemm_async(const half* A, const half* B, float* C, ...) {
    __shared__ __align__(32) half Ashared[2][BLOCK_M][BLOCK_K+4];
    __shared__ __align__(32) half Bshared[2][BLOCK_K][BLOCK_N+4];
    
    nvcuda::wmma::fragment<...> a_frag, b_frag, acc_frag;
    pipeline pipe;
    
    // 流水线执行
    for(int k=0; k<K; k+=BLOCK_K){
        // 阶段1：异步加载下一块
        if(k+BLOCK_K < K) {
            load_tile_async(A_next, Ashared[next_buf], ...);
            load_tile_async(B_next, Bshared[next_buf], ...);
        }
        
        // 阶段2：TensorCore计算
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        
        // 阶段3：缓冲区切换
        __syncthreads();
        swap(current_buf, next_buf);
    }
}
```

### 3.2 性能分析指南
使用Nsight Systems生成性能画像：
```bash
nsys profile --stats=true --trace=cuda,nvtx --cuda-memory-usage=true ./hgemm  3072 3072 3072 data
```
需关注的指标：
- `sm__throughput.avg.pct_of_peak_sustained`: TensorCore利用率
- `dram__throughput.avg.pct_of_peak_sustained`: 显存带宽
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.avg.pct_of_peak_sustained`: 全局加载效率

你将看到类似内容：
```txt
Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)           Name         
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------
     65.7        190767219          3  63589073.0     98885.0     91262  190577072  109974833.2  cudaMalloc            
     19.7         57189533          1  57189533.0  57189533.0  57189533   57189533          0.0  cudaEventSynchronize  
     14.0         40756856          3  13585618.7   8174435.0   4059456   28522965   13098721.3  cudaMemcpy            
      0.4          1300395          3    433465.0    522777.0    252472     525146     156749.0  cudaFree              
      0.1           187795          1    187795.0    187795.0    187795     187795          0.0  cudaLaunchKernel      
      0.0            18556          2      9278.0      9278.0      4745      13811       6410.6  cudaEventRecord       
      0.0            15571          2      7785.5      7785.5       441      15130      10386.7  cudaEventCreate       
      0.0              934          1       934.0       934.0       934        934          0.0  cuModuleGetLoadingMode
```

## 四、提交内容
1. **代码包**：
将您的代码打包，其内部文件夹类似如下结构：
   ```
   HPC_TeamX/
   ├── src/
   │   ├── hgemm.cu           # 主实现
   │   └── hgemm_utils.cuh    # 辅助函数
   ├── Makefile
   ├── README.md  # 如何编译+运行的简单说明
   └── Otherfiles # 其他辅助文件如 setup.py
   ```


2. **技术报告**（PDF格式）：
   - 优化策略示意图（推荐使用TiKZ或Draw.io绘制）
   - Nsight性能截图（需包含时间轴和指标表格）
   - 不同案例的性能对比图

3. **测试脚本**：
   ```bash
   # 示例测试脚本
   for case in 1 2 3 4 5; do
       ./hgemm -m ${M[$case]} -n ${N[$case]} -k ${K[$case]} \
               -o result_${case}.bin
   done
   ```

## 五、编译与验证
### 5.1 编译指令
```bash
# 基础编译
nvcc -arch=sm_70 -Xcompiler -fopenmp -O3 \
     -I./include -L${CUDA_PATH}/lib64 \
     -lcublas -o hgemm hgemm.cu

# 带Nsight调试的版本
nvcc -lineinfo -src-in-ptx -keep -G \
     -arch=sm_70 -o hgemm_debug hgemm.cu
```

### 5.2 正确性验证
```cpp
// 使用cuBLAS作为基准
cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha,
            dA, M, dB, K, &beta, dC, M);

// 计算相对误差
float max_err = 0;
for(int i=0; i<M*N; ++i){
    float ref = cublas_result[i];
    float val = your_result[i];
    max_err = fmaxf(max_err, fabsf(val-ref)/fabsf(ref));
}
```

## 六、参考文献
1.  NVIDIA Tensor Core 编程指南 v1.3
2.  CUDA C++ Programming Guide - Asynchronous Copy
3.  Volta架构白皮书 - 第4章 Tensor Core设计
4.  "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking" - IEEE Access 2020
5.  上科大HPC教程
6. Azzam Haidar, Stanimire Tomov, Jack Dongarra, and Nicholas J. Higham. 2018. Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision iterative refinement solvers. In Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC '18). IEEE Press, Article 47, 1–11.

