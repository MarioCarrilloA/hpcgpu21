EXERCISE_NUM = 4

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
LDLIBS = -lm
TARGET = exercise0$(EXERCISE_NUM)
SOURCES = main.cu
DEVINFO_BIN = DevProperties

all:
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES) -o $(TARGET)


extras:
	$(NVCC) $(CUDAFLAGS) gpu_basics.cpp -o $(DEVINFO_BIN)
