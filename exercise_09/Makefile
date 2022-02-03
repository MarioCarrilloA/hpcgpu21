EXERCISE_NUM = 9

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
LDLIBS = -lm
TARGET = exercise0$(EXERCISE_NUM)
SOURCES = main.cu

all:
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES) -o $(TARGET)

