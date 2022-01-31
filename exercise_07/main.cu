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
    float maxrad = 0.9f;
    float f = 0.0;
    float dsq;
    float distx, disty, distz;
    int size = blockDim.x * 2;

    extern __shared__ float x_shared[];
    p xi, xj;
    xi.x = &x_shared[0];
    xi.y = &x_shared[blockDim.x];
    xi.z = &x_shared[blockDim.x * 2];
    xi.m = &x_shared[blockDim.x * 3];

    xj.x = &x_shared[blockDim.x * 4];
    xj.y = &x_shared[blockDim.x * 6];
    xj.z = &x_shared[blockDim.x * 8];
    xj.m = &x_shared[blockDim.x * 10];

    if (i < npart) {
        *((float*)(&xi.x[0]) + t) = *((float*)(&xin.x[g * blockDim.x]) + t);
        *((float*)(&xi.y[0]) + t) = *((float*)(&xin.y[g * blockDim.x]) + t);
        *((float*)(&xi.z[0]) + t) = *((float*)(&xin.z[g * blockDim.x]) + t);
        *((float*)(&xi.m[0]) + t) = *((float*)(&xin.m[g * blockDim.x]) + t);

        for(int ja = 0; ja < npart; ja+= size) {
            for(int jl = 0; jl < size; jl += blockDim.x){
                *((float*)(&xj.x[jl]) + t) = *((float*)(&xin.x[ja + jl]) + t);
                *((float*)(&xj.y[jl]) + t) = *((float*)(&xin.y[ja + jl]) + t);
                *((float*)(&xj.z[jl]) + t) = *((float*)(&xin.z[ja + jl]) + t);
                *((float*)(&xj.m[jl]) + t) = *((float*)(&xin.m[ja + jl]) + t);
            }
            __syncthreads();

            for(int j = ja; j < ja + size; j++){
                distx = xi.x[t] - xj.x[j - ja];
                disty = xi.y[t] - xj.y[j - ja];
                distz = xi.z[t] - xj.z[j - ja];

                dsq = distx * distx + disty * disty + distz * distz;

                if (dsq < maxrad && dsq != 0 && i != j) {
                    f += xi.m[t] * xj.m[j - ja] * (xi.x[t] - xj.x[j - ja]) / dsq;
                }
            }
        }
        double s = f * dt * val;
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


int exercise04(int npart, int niters) {
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
    exercise04(npart, niters);

    return 0;
}

