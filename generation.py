import numpy as np

def generate_matrix_file(filename, M=2560, N=2560, K=2560):
    # 设置随机种子以确保可重复性
    np.random.seed(42)
    
    # 生成矩阵A (M×K) 和 B (K×N) 的随机值
    A = np.random.uniform(0.0, 1.0, (M, K)).astype(np.float16)
    B = np.random.uniform(0.0, 1.0, (K, N)).astype(np.float16)
    
    # 写入文件
    with open(filename, 'w') as f:
        # 写入M, N, K
        f.write(f"{M} {N} {K}\n")
        
        # 写入矩阵A（行优先）
        for i in range(M):
            for j in range(K):
                f.write(f"{float(A[i, j]):.6f}\n")
                
        # 写入矩阵B（行优先）
        for i in range(K):
            for j in range(N):
                f.write(f"{float(B[i, j]):.6f}\n")

if __name__ == "__main__":
    M = 4096
    N = 4096
    K = 4096

    # 生成矩阵文件
    generate_matrix_file("matrices.txt", M, N, K)