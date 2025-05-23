# 结果验证脚本（与cuBLAS对比）
import numpy as np

def read_bin(file, shape, dtype):
    return np.fromfile(file, dtype=dtype).reshape(shape)

M, N, K = 512, 1024, 768
A = read_bin("data/A.bin", (M, K), np.float16)
B = read_bin("data/B.bin", (K, N), np.float16)
C_ref = np.dot(A.astype(np.float32), B.astype(np.float32))
C_gpu = read_bin(f"result_{M}_{N}_{K}.bin", (M, N), np.float32)

print("最大相对误差:", np.max(np.abs(C_gpu - C_ref) / np.abs(C_ref).max()))
# 输出性能
print("性能:", M * N * K * 2 / (C_gpu.nbytes / 1e9) / 1e6, "GFLOPS")
