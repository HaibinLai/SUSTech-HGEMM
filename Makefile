# compiling hgemm.cu with nvcc

NVCC = nvcc

# flags
NVCC_FLAGS = -arch=sm_70 -O3

SRC_DIR = src
BLAS_SRC = $(SRC_DIR)/hgemm_cublas.cu
BENCH_SRC = $(SRC_DIR)/hgemm_cublas_bench.cu

TARGET_CUBLAS = hgemm_cublas
TARGET_CUBLAS_BENCH = hgemm_cublas_bench

# Default target
all: $(TARGET_CUBLAS) $(TARGET_CUBLAS_BENCH)

# Create build directory
build:
	mkdir -p build

$(TARGET_CUBLAS): build $(BLAS_SRC)
	$(NVCC) $(NVCC_FLAGS) $(BLAS_SRC) -o build/$(TARGET_CUBLAS) -lcublas -lcudart

$(TARGET_CUBLAS_BENCH): build $(BENCH_SRC)
	$(NVCC) $(NVCC_FLAGS) $(BENCH_SRC) -o build/$(TARGET_CUBLAS_BENCH) -lcublas -lcudart

# Clean up
clean:
	rm -f build/$(TARGET_CUBLAS)
	rm -f build/$(TARGET_CUBLAS_BENCH)

# Phony targets
.PHONY: all clean