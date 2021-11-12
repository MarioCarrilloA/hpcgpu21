EXERCISE_NUM = 2

# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
LDLIBS = -lGLEW -lglut -lGLU -lGL -lm
TARGET = exercise0$(EXERCISE_NUM)
SOURCES = sim_skel.cu optics.cpp
DEVINFO_BIN = DevProperties


all:
	$(NVCC) $(CUDAFLAGS) $(LDFLAGS) $(LDLIBS) $(SOURCES) -o $(TARGET)

extras:
	$(NVCC) $(CUDAFLAGS) gpu_basics.cpp -o $(DEVINFO_BIN)

