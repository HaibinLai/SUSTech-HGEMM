import numpy as np

def generate_matrix_file(filename, M=2560, N=2560, K=2560):
    # random seed for reproducibility
    np.random.seed(42)

    # generate random values for matrix A (M×K) and B (K×N)
    A = np.random.uniform(0.0, 1.0, (M, K)).astype(np.float16)
    B = np.random.uniform(0.0, 1.0, (K, N)).astype(np.float16)

    # write to file
    with open(filename, 'w') as f:
        # write M, N, K
        f.write(f"{M} {N} {K}\n")

        # write matrix A (row-major)
        for i in range(M):
            for j in range(K):
                f.write(f"{float(A[i, j]):.6f}\n")

        # write matrix B (row-major)
        for i in range(K):
            for j in range(N):
                f.write(f"{float(B[i, j]):.6f}\n")

if __name__ == "__main__":
    M = 4096
    N = 4096
    K = 4096

    # generate matrix file
    generate_matrix_file("matrices.txt", M, N, K)