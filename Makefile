# compiling hgemm.cu with nvcc

NVCC = nvcc

# flags
NVCC_FLAGS = -arch=sm_70 -O3

SRC_DIR = src

BLAS_SRC = $(SRC_DIR)/hgemm_cublas.cu
CUSTOM_SRC = $(SRC_DIR)/hgemm.cu
BENCH_SRC = $(SRC_DIR)/hgemm_cublas_bench.cu
COMPARE_SRC = $(SRC_DIR)/hgemm_compare.cu

TARGET_CUBLAS = hgemm_cublas
TARGET_CUSTOM = hgemm_custom
TARGET_CUBLAS_BENCH = hgemm_cublas_bench
TARGET_COMPARE = hgemm_compare

# Default target
all: $(TARGET_CUBLAS) $(TARGET_CUBLAS_BENCH) $(TARGET_COMPARE) $(TARGET_CUSTOM)

# Create build directory
build:
	mkdir -p build

$(TARGET_CUBLAS): build $(BLAS_SRC)
	$(NVCC) $(NVCC_FLAGS) $(BLAS_SRC) -o build/$(TARGET_CUBLAS) -lcublas -lcudart

$(TARGET_CUBLAS_BENCH): build $(BENCH_SRC)
	$(NVCC) $(NVCC_FLAGS) $(BENCH_SRC) -o build/$(TARGET_CUBLAS_BENCH) -lcublas -lcudart

$(TARGET_COMPARE): build $(COMPARE_SRC)
	$(NVCC) $(NVCC_FLAGS) $(COMPARE_SRC) -o build/$(TARGET_COMPARE) -lcublas -lcudart

$(TARGET_CUSTOM): build $(CUSTOM_SRC)
	$(NVCC) $(NVCC_FLAGS) $(CUSTOM_SRC) -o build/$(TARGET_CUSTOM) -lcudart

# Clean up
clean:
	rm -f build/$(TARGET_CUBLAS)
	rm -f build/$(TARGET_CUBLAS_BENCH)
	rm -f build/$(TARGET_COMPARE)
	rm -f build/$(TARGET_CUSTOM)

# Phony targets
.PHONY: all clean