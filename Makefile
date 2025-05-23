# compiling hgemm.cu with nvcc

NVCC = nvcc

# flags
NVCC_FLAGS = -arch=sm_70 -O3

SRC_DIR = src
SRC = $(SRC_DIR)/hgemm.cu


TARGET = hgemm_GPU

# Default target
all: $(TARGET)

# build
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

# Clean up
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean