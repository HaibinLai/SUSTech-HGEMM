# 南方科技大学 HPC 校内赛 - GPU-HGEMM 加速赛题

**联系人**：赖海斌 12211612@mail.sustech.edu.cn  
**硬件平台**：NVIDIA V100 GPU (32GB显存) × 1 + Xeon Platinum CPU

为了帮助大家快速上手，我给大家准备了这个简单的 repo，省去在 IO 上编程的时间。

## 项目结构

```
SUSTech-HGEMM/
├── Makefile                 # 构建脚本
├── README.md               # 项目说明
├── src/                    # 源代码目录
│   ├── hgemm_cublas.cu     # cuBLAS 实现
│   ├── hgemm.cu            # 自定义 HGEMM 实现
│   └── ...
├── data/                   # 数据目录
│   ├── input/              # 输入矩阵数据
│   │   ├── Case1_768x768x768/
│   │   ├── Case2_128x1024x2048/
│   │   └── ...
│   └── output/             # 输出结果
├── scripts/                # 运行脚本
│   ├── generation.sh       # 数据生成脚本
│   ├── run_example.sh      # 运行示例
│   ├── cuBLAS_benchmark.sh # cuBLAS 性能测试
│   └── ...
├── tools/                  # 工具脚本
│   ├── matrix_generation.py # 矩阵生成工具
│   └── comparsion.py       # 结果比较工具
└── build/                  # 构建输出目录
```

## 快速开始

### 1. 环境要求

- NVIDIA GPU (支持 CUDA Compute Capability 7.0 或更高)
- CUDA Toolkit (>=9.0)
- cuBLAS 库
- Python 3.x (用于数据生成和分析)


**初始化：**

```bash
bash ./scripts/init_checking.sh
```

### 2. 数据生成

使用工具脚本生成测试矩阵数据：

```bash
./scripts/generation.sh
```

其会调用 `/tools/matrix_generation.py` , 您也可以直接使用该脚本生成想要生成的数据：

```bash
python ./tools/matrix_generation.py
```


生成过程大约需要 3-8 分钟，会创建 10 个不同尺寸的测试案例：

- Case1: 768×768×768
- Case2: 128×1024×2048
- Case3: 128×2048×8192
- Case4: 512×3072×1024
- Case5: 512×4096×8192
- Case6: 3136×576×64
- Case7: 4096×4096×4096
- Case8: 1024×16384×16384
- Case9: 4096×16384×14336
- Case10: 32768×32768×32768

### 3. 编译

使用 Makefile 进行编译：

```bash
cd SUSTech-HGEMM
make
```

这会编译以下目标：
- `hgemm_cublas`: cuBLAS 实现
- `hgemm_custom`: 您的 HGEMM 实现
- `hgemm_cublas_bench`: cuBLAS 性能测试
- `hgemm_compare`: 结果比较工具（您可以将您的kernel与cuBLAS进行性能比较）

注意：Makefile 中使用了 `sm_70` 架构（适用于 V100）。如果使用其他 GPU，请修改相应的架构参数。

### 4. 运行

推荐使用 `scripts/run_example.sh` 进行测试

```bash
bash ./scripts/run_example.sh
```

#### 运行 cuBLAS 版本：

```bash
./build/hgemm_cublas -d data/input/Case1_768x768x768 -o data/output
```

#### 运行自定义实现：

```bash
./build/hgemm_custom -d data/input/Case1_768x768x768 -o data/output
```

#### 批量测试：

```bash
./scripts/run_example.sh
```

## 性能评估

### 基准测试

运行 cuBLAS 基准测试：

```bash
./scripts/cuBLAS_benchmark.sh
```

运行 10 次平均基准测试：

```bash
./scripts/cuBLAS_benchmark_avg10.sh
```

### 结果比较

比较不同实现的结果：

```bash
./scripts/compare.sh
```


### profile

`./scripts/custom_profile.sh` 使用了ncu对特定的输入进行测试。

```bash
bash ./scripts/custom_profile.sh
```



### 绘图

`tools/draw_figure.ipynb` 提供了很好的log分析及作图脚本。

运行以下脚本，将输出导入 `benchmark.log`
```bash
./scripts/cuBLAS_benchmark.sh > benchmark.log
```

随后可对此进行绘图分析:

<!-- 展示图片 -->
![benchmark](fig/benchmark_results.png)




## 清理

删除编译产生的文件：

```bash
make clean
```


## 精度比较

在本次赛题中，精度比较的分数将由以下函数确定，您可在`src/hgemm_compare.cu` 中找到。

```CPP
// 计算Frobenius范数的相对误差
__global__ void compute_fp16_relative_error_kernel(const __half* ref, const __half* calc,
                                                   float* diff_sq, float* ref_sq, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float ref_val = __half2float(ref[idx]);
        float calc_val = __half2float(calc[idx]);
        float diff = calc_val - ref_val;
        diff_sq[idx] = diff * diff;  // 计算差的平方
        ref_sq[idx] = ref_val * ref_val;  // 计算参考值的平方
    }
}

// GPU接口计算相对误差
float compute_relative_error_fp16_gpu(const __half* d_ref, const __half* d_calc, int size) {
    float *d_diff_sq, *d_ref_sq;
    cudaMalloc(&d_diff_sq, size * sizeof(float));
    cudaMalloc(&d_ref_sq, size * sizeof(float));

    int block = 256;
    int grid = (size + block - 1) / block;
    compute_fp16_relative_error_kernel<<<grid, block>>>(d_ref, d_calc, d_diff_sq, d_ref_sq, size);
    cudaDeviceSynchronize();

    std::vector<float> h_diff_sq(size);
    std::vector<float> h_ref_sq(size);
    cudaMemcpy(h_diff_sq.data(), d_diff_sq, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref_sq.data(), d_ref_sq, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_diff_sq);
    cudaFree(d_ref_sq);

    float sum_diff = 0.f, sum_ref = 0.f;
    for (int i = 0; i < size; ++i) {
        sum_diff += h_diff_sq[i];
        sum_ref += h_ref_sq[i];
    }

    return std::sqrt(sum_diff / sum_ref);
}
```



## 问题反馈

如有问题请联系：12211612@mail.sustech.edu.cn



