import numpy as np
from tqdm import tqdm
import sys
import argparse

def generate_matrix_file(filename, M=4096, N=4096, K=4096):
    np.random.seed(42)
    A = np.random.uniform(0.0, 1.0, (M, K)).astype(np.float16)
    B = np.random.uniform(0.0, 1.0, (K, N)).astype(np.float16)

    with open(filename, 'w') as f:
        f.write(f"{M} {N} {K}\n")
        for i in tqdm(range(M), desc="Writing A", file=sys.stderr):
            for j in range(K):
                f.write(f"{float(A[i, j]):.6f}\n")
        for i in tqdm(range(K), desc="Writing B", file=sys.stderr):
            for j in range(N):
                f.write(f"{float(B[i, j]):.6f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate matrix file with dimensions M, N, K")
    parser.add_argument("--M", type=int, default=4096, help="Number of rows of matrix A and C")
    parser.add_argument("--N", type=int, default=4096, help="Number of columns of matrix B and C")
    parser.add_argument("--K", type=int, default=4096, help="Inner dimension of matrices A and B")
    parser.add_argument("--output", type=str, default="data/input/matrices.txt", help="Output filename")

    args = parser.parse_args()

    generate_matrix_file(args.output, args.M, args.N, args.K)
