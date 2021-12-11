# Nvidia compiler
NVCC = nvcc
CUDAFLAGS = -O -I${CUDA_INSTALL_PATH}/samples/common/inc
LDFLAGS = -L${CUDA_INSTALL_PATH}/samples/common/lib/linux/x86_64
LDLIBS = -lm -lcublas -lcurand
BENCH_BIN = benchmark

all:
	$(NVCC) $(CUDAFLAGS) CopyBench.cpp -o $(BENCH_BIN)
