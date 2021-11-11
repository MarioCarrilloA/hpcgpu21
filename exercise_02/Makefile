EXERCISE_NUM = 2

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
DEVINFO_BIN = DevProperties

extras:
	$(NVCC) $(CUDAFLAGS) gpu_basics.cpp -o $(DEVINFO_BIN)

