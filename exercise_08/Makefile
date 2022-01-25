EXERCISE_NUM = 8

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -I${CUDA_INSTALL_PATH}/samples/common/inc -std=c++11
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
LDLIBS = -lcublas -lcurand
TARGET = exercise0$(EXERCISE_NUM)
SOURCES = main.cu

all:
	@$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)


