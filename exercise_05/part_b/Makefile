EXERCISE_NUM = 5

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O3 -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64 -arch=compute_70 -code=sm_70
LDLIBS = -lm -lcublas -lcurand
TARGET = exercise0$(EXERCISE_NUM)
SOURCES_A = main_a.cu
SOURCES_B = main_b.cu

all:
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES_A) -o $(TARGET)_b1
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES_B) -o $(TARGET)_b2

b1: 
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES_A) -o $(TARGET)_b1

b2: 
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES_B) -o $(TARGET)_b2