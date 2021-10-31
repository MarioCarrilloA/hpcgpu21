EXERCISE_NUM = 0

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc
TARGET = exercise0$(EXERCISE_NUM)

all:
	$(NVCC) $(CUDAFLAGS) gpu_basics.cpp -o $(TARGET)
