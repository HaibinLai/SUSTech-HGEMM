import numpy as np
# from tqdm import tqdm
import sys
import argparse
import os
import multiprocessing
import struct

def write_matrix_A_bin(filename, M, K):
    np.random.seed(42)
    A = np.random.uniform(0.0, 1.0, (M, K)).astype(np.float16)
    with open(filename, 'wb') as fa:
        fa.write(struct.pack("ii", M, K))  # 写入维度（两个 int32）
        fa.write(A.tobytes())

def write_matrix_B_bin(filename, K, N):
    np.random.seed(43)
    B = np.random.uniform(0.0, 1.0, (K, N)).astype(np.float16)
    with open(filename, 'wb') as fb:
        fb.write(struct.pack("ii", K, N))
        fb.write(B.tobytes())

def generate_matrices_parallel(outdir, M, N, K):
    os.makedirs(outdir, exist_ok=True)

    A_filename = os.path.join(outdir, "A_matrix.bin")
    B_filename = os.path.join(outdir, "B_matrix.bin")

    p1 = multiprocessing.Process(target=write_matrix_A_bin, args=(A_filename, M, K))
    p2 = multiprocessing.Process(target=write_matrix_B_bin, args=(B_filename, K, N))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print(f"✅ Generated binary matrices:")
    print(f"   A: ({M}, {K}) -> {A_filename}")
    print(f"   B: ({K}, {N}) -> {B_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary A and B matrix files using multiprocessing")
    parser.add_argument("--M", type=int, default=4096, help="Number of rows of matrix A")
    parser.add_argument("--N", type=int, default=4096, help="Number of columns of matrix B")
    parser.add_argument("--K", type=int, default=4096, help="Inner dimension shared by A and B")
    parser.add_argument("--outdir", type=str, default="data/input/matrix_case", help="Output directory for matrix files")

    args = parser.parse_args()
    generate_matrices_parallel(args.outdir, args.M, args.N, args.K)
