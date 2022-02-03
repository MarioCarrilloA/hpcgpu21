EXERCISE_NUM = 9

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc -I./ -std=c++11
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
LDLIBS = -lm
TARGET = exercise0$(EXERCISE_NUM)
SOURCES = main.cu kernel1.cu kernel2.cu

all:
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES) -o $(TARGET)

