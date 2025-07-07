# compiling hgemm.cu with nvcc

NVCC = nvcc

# flags
NVCC_FLAGS = -arch=sm_70 -O3

SRC_DIR = src
BLAS_SRC = $(SRC_DIR)/hgemm_cublas.cu

TARGET_CUBLAS = hgemm_cublas

# Default target
all: $(TARGET_CUBLAS)

# build
$(TARGET_CUBLAS): $(BLAS_SRC)
	$(NVCC) $(NVCC_FLAGS) $(BLAS_SRC) -o build/$(TARGET_CUBLAS) -lcublas -lcudart

# Clean up
clean:
	rm -f build/$(TARGET_CUBLAS)

# Phony targets
.PHONY: all clean