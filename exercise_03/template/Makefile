EXERCISE_NUM = 3

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
LDLIBS = -lGLEW -lglut -lGLU -lGL -lm
TARGET = exercise0$(EXERCISE_NUM)
SOURCESP1 = exercise03_p1.cpp  kernel_p1.cu
SOURCESP2 = exercise03_p2.cpp  kernel_p2.cu
DEVINFO_BIN = DevProperties

all:
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCESP1) -o $(TARGET)_p1
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCESP2) -o $(TARGET)_p2

extras:
	$(NVCC) $(CUDAFLAGS) gpu_basics.cpp -o $(DEVINFO_BIN)
