# 南方科技大学 HPC 校内赛 - GPU-HGEMM 加速赛题

**联系人**：赖海斌 12211612@mail.sustech.edu.cn  
**硬件平台**：NVIDIA V100 GPU (32GB显存) \* 1 + Xeon Platinum CPU

为了帮助大家快速上手，我给大家准备了这个简单的 repo ，省去在io上编程的时间。


## 生成矩阵数据

请使用 tools 文件夹下生成的python脚本

```python
python ./tools/generation.py
```

脚本使用两个进程来生成两个输入矩阵（我不太确定用一个进程是否更快，毕竟io速度在那）。生成全部的矩阵案例大概需要10-15分钟。

生成演示：

```bash
✅ Generated matrices with dimensions:
   A: (4096, 4096) -> data/input/Case7_4096x4096x4096/A_matrix.txt
   B: (4096, 4096) -> data/input/Case7_4096x4096x4096/B_matrix.txt
📦 Generating matrices in data/input/Case8_1024x16384x16384 (M=1024, N=16384, K=16384)
Writing A: 100%|██████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:08<00:00, 120.89it/s]
Writing B: 100%|████████████████████████████████████████████████████████████████████████████| 16384/16384 [02:18<00:00, 118.55it/s]
✅ Generated matrices with dimensions:
   A: (1024, 16384) -> data/input/Case8_1024x16384x16384/A_matrix.txt
   B: (16384, 16384) -> data/input/Case8_1024x16384x16384/B_matrix.txt
📦 Generating matrices in data/input/Case9_4096x16384x14336 (M=4096, N=16384, K=14336)
Writing A: 100%|██████████████████████████████████████████████████████████████████████████████| 4096/4096 [00:29<00:00, 137.90it/s]
Writing B: 100%|████████████████████████████████████████████████████████████████████████████| 14336/14336 [01:52<00:00, 126.91it/s
```

## 编译

由于程序简单，我们就直接使用make进行编译：

```bash
cd SUSTech-HGEMM
make
```

注意在make里我使用了sm_70架构。详细的cuda架构与对应参数可以查看这篇文章：https://zhuanlan.zhihu.com/p/631850036

当你想删除你编译出的二进制结果时，请使用

```bash
make clean
```


## 运行



## 与 cublas 比较结果


## Profile



