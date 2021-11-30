EXERCISE_NUM = 3
SUFFIX = part_b

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
LDLIBS = -lGLEW -lglut -lGLU -lGL -lm
TARGET = exercise$(EXERCISE_NUM)_$(SUFFIX)
SOURCES = main.cpp  kernel.cu

all:
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES) -o $(TARGET)

