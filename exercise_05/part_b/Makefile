EXERCISE_NUM = 5

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O3 -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64 -arch=compute_70 -code=sm_70
LDLIBS = -lm -lcublas -lcurand
TARGET = exercise0$(EXERCISE_NUM)
SOURCES = main.cu

all:
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES) -o $(TARGET)_b1
