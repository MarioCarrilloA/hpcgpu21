#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <math.h>
#include <unistd.h>
#include <helper_cuda.h>
#include <sys/time.h>

#define DEFAULT_NUM_ITERATIONS 1000
#define DEFAULT_NUM_PARTICLES  80000
#define DEFAULT_NUM_TO_SHOW    10
#define MAX_THREADS_PER_BLOCK 1024

using namespace std;

struct p {
    float *x;
    float *y;
    float *z;
    float *m;
};

static const char help[] =
    "Usage: exercise07 [-k number] [-i number] [-p number] [-h]\n"
    "Description:\n"
    "  -i number:     Specifies how many times the kernel will be\n"
    "                 executed.\n"
    "  -p number:     Number of particles to be processed\n"
    "  -h             Prints this help message.\n";

void Print(p x) {
    for (int i = 0; i < DEFAULT_NUM_TO_SHOW; i++)
        cout << x.x[i] << endl;
}

void init(p xin, long npart) {
    for (int i = 0; i < npart; i++) {
        xin.x[i] = (float(rand())/float((RAND_MAX)) * 10.0f) + 0.1f;
        xin.y[i] = (float(rand())/float((RAND_MAX)) * 10.0f) + 0.1f;
        xin.z[i] = (float(rand())/float((RAND_MAX)) * 10.0f) + 0.1f;
        xin.m[i] = (float(rand())/float((RAND_MAX)) * 10.0f) + 0.1f;
    }
}

// GPU Kernel
__global__ void kernel(p xin, p xout, long int npart, double dt, double val) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int m = 2;
    int size = m * blockDim.x;
    float maxrad = 0.9f;
    float f = 0.0;

    // distance vars
    float dsq;
    float dstx;
    float dsty;
    float dstz;

    extern __shared__ float x_shared[];
    p xj_shared;

    // Split shared memry
    xj_shared.x = &x_shared[0];
    xj_shared.y = &x_shared[blockDim.x * 2];
    xj_shared.z = &x_shared[blockDim.x * 4];
    xj_shared.m = &x_shared[blockDim.x * 8];

    if (i < npart) {
        xout.x[i] = xin.x[i];
        xout.y[i] = xin.y[i];
        xout.z[i] = xin.z[i];

        for(int ja = 0; ja < npart; ja+=size) {

            // compute particles according to block size multiplier
            for(int jl = 0; jl < size/blockDim.x; jl += blockDim.x) {
                int jdx = jl + t;
                int idx = ja + jl + t;

                // Copy to shared memory
                xj_shared.x[jdx] = xin.x[idx];
                xj_shared.y[jdx] = xin.y[idx];
                xj_shared.z[jdx] = xin.z[idx];
                xj_shared.m[jdx] = xin.m[idx];
            }
            __syncthreads();

            for(int j = ja; j < ja + size; j++){
                dstx = xin.x[t] - xj_shared.x[j - ja];
                dsty = xin.y[t] - xj_shared.y[j - ja];
                dstz = xin.z[t] - xj_shared.z[j - ja];

                // Compute distance
                dsq = (dstx * dstx) + (dsty * dsty) + (dstz * dstz);

                if (dsq < maxrad && dsq != 0 && i != j) {
                    f += xin.m[t] * xj_shared.m[j - ja] * (xin.x[t] - xj_shared.x[j - ja]) / dsq;
                }
            }
        }

        float s = f * dt * val;
        xout.x[i] += s;
        xout.y[i] += s;
        xout.z[i] += s;
    }
}


void execute_kernel(p xin, p xout, int npart, int niters) {
    p x_dev;
    p xin_dev;
    p xout_dev;
    float dt = 0.5f;
    float val = 0.5f;
    float execution_time = 0.0f;

    // Structures to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Calculate blocks & threads
    int minimum_blocks;
    int minimum_threads;
    int extra_block;
    int total_blocks;

    // Minimum number of blocks/threads calculation
    if (npart < MAX_THREADS_PER_BLOCK) {
        minimum_blocks = 1;
        minimum_threads = npart;
    } else {
        minimum_blocks = npart / MAX_THREADS_PER_BLOCK;
        minimum_threads = MAX_THREADS_PER_BLOCK;
    }

    // Extra block calculation
    if (npart % MAX_THREADS_PER_BLOCK == 0 || npart < MAX_THREADS_PER_BLOCK)
        extra_block = 0;
    else
        extra_block = 1;

    total_blocks = minimum_blocks + extra_block;
    dim3 blocks(total_blocks, 1, 1);
    dim3 threads(minimum_threads, 1, 1);
    printf("Blocks:%d   Threads:%d\n", total_blocks, minimum_threads);
    printf( "Executing ...\n");

    // START measure time
    cudaEventRecord(start, 0);

    // Memory management
    checkCudaErrors(cudaMalloc((void **)&xin_dev.x, sizeof(float) * npart));
    checkCudaErrors(cudaMalloc((void **)&xin_dev.y, sizeof(float) * npart));
    checkCudaErrors(cudaMalloc((void **)&xin_dev.z, sizeof(float) * npart));
    checkCudaErrors(cudaMalloc((void **)&xin_dev.m, sizeof(float) * npart));
    checkCudaErrors(cudaMalloc((void **)&xout_dev.x, sizeof(float) * npart));
    checkCudaErrors(cudaMalloc((void **)&xout_dev.y, sizeof(float) * npart));
    checkCudaErrors(cudaMalloc((void **)&xout_dev.z, sizeof(float) * npart));
    checkCudaErrors(cudaMalloc((void **)&xout_dev.m, sizeof(float) * npart));
    checkCudaErrors(cudaMemcpy(xin_dev.x, xin.x, sizeof(float) * npart, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(xin_dev.y, xin.y, sizeof(float) * npart, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(xin_dev.z, xin.z, sizeof(float) * npart, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(xin_dev.m, xin.m, sizeof(float) * npart, cudaMemcpyHostToDevice));

    // for dynamic allocation of shared memory
    // Kernel 1 execution
    for (int i = 0; i < niters; i++) {
        kernel<<<blocks, threads, sizeof(float) * 1024 * 12>>>(xin_dev, xout_dev, npart, dt, val);

        // Exchange pointers
        x_dev = xin_dev;
        xin_dev = xout_dev;
        xout_dev = x_dev;
    }

    checkCudaErrors(cudaMemcpy(xout.x, xout_dev.x, sizeof(float) * npart, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(xout.y, xout_dev.y, sizeof(float) * npart, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(xout.z, xout_dev.z, sizeof(float) * npart, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(xout.m, xout_dev.m, sizeof(float) * npart, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(xin_dev.x));
    checkCudaErrors(cudaFree(xin_dev.y));
    checkCudaErrors(cudaFree(xin_dev.z));
    checkCudaErrors(cudaFree(xin_dev.m));
    checkCudaErrors(cudaFree(xout_dev.x));
    checkCudaErrors(cudaFree(xout_dev.y));
    checkCudaErrors(cudaFree(xout_dev.z));
    checkCudaErrors(cudaFree(xout_dev.m));

    // STOP measure time
    cudaEventRecord(stop, 0);

    //Print(xout);

    // Calculate time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&execution_time, start, stop);
    printf("kernel execution: %f seconds\n", (execution_time / 1000.0f));
}


int exercise07(int npart, int niters) {
    p xin, xout;

    xin.x = (float*)malloc(sizeof(float) * npart);
    xin.y = (float*)malloc(sizeof(float) * npart);
    xin.z = (float*)malloc(sizeof(float) * npart);
    xin.m = (float*)malloc(sizeof(float) * npart);

    xout.x = (float*)malloc(sizeof(float) * npart);
    xout.y = (float*)malloc(sizeof(float) * npart);
    xout.z = (float*)malloc(sizeof(float) * npart);
    xout.m = (float*)malloc(sizeof(float) * npart);
    init(xin, npart);
    execute_kernel(xin, xout, npart, niters);
    free(xin.x);
    free(xin.y);
    free(xin.z);
    free(xin.m);
    free(xout.x);
    free(xout.y);
    free(xout.z);
    free(xout.m);

    return 0;
}


int main(int argc, char **argv) {
    int opt;
    int niters = DEFAULT_NUM_ITERATIONS;
    long int npart = DEFAULT_NUM_PARTICLES;

    while ((opt = getopt(argc, argv, "i:p:h")) != EOF) {
        switch (opt) {
            case 'i':
                niters = atoi(optarg);
                break;
            case 'p':
                npart = atoi(optarg);
                break;
            case 'h':
                cout << help << endl;
                return 0;
            case '?':
                cerr << "error: unknown option" << endl;
                cout << help << endl;
                return 1;
            default:
                cerr << help << endl;
                return 1;
        }
    }

    cout << "Particles: " << npart << endl;
    cout << "Iterations: " << niters << endl;
    exercise07(npart, niters);

    return 0;
}

