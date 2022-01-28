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

// This number was obtained by checking the device properties
// of a P2000 card. This project includes a script to use it.
#define MAX_THREADS_PER_BLOCK 1024

using namespace std;

static const char help[] =
    "Usage: exercise04 [-k number] [-i number] [-p number] [-h]\n"
    "Description:\n"
    "  -i number:     Specifies how many times the kernel will be\n"
    "                 executed.\n"
    "  -p number:     Number of particles to be processed\n"
    "  -h             Prints this help message.\n";

void Print(float *arr) {
    for (int i = 0; i < DEFAULT_NUM_TO_SHOW; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

void init(float *arr, long npart) {
    for (int i = 0; i < npart; i++) {
        arr[i] = (float(rand())/float((RAND_MAX)));
    }
}

// GPU Kernel
__global__ void kernel(float *xin, float *yin, float *zin,
                        float *xout, float *yout, float *zout,
                        float *m, long int npart, double dt, double val) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    float maxrad = 0.9f;
    float f = 0.0f;

    // distance vars
    float dsq;
    float dstx;
    float dsty;
    float dstz;

    if (i < npart) {
        xout[i] = xin[i];
        yout[i] = yin[i];
        zout[i] = zin[i];

        for (int j = 0; j < npart; j++) {
            dstx = xin[i] - xin[j];
            dsty = yin[i] - yin[j];
            dstz = zin[i] - zin[j];
            dsq =  (dstx * dstx) + (dsty * dsty) + (dstz * dstz);

            if (dsq < maxrad && dsq != 0 && i != j) {
                f += m[i] * m[j] * (xin[i] - xin[j]) / dsq;
            }
        }

        float s = f * dt * val;
        xout[i] += s;
        yout[i] += s;
        zout[i] += s;
    }
}


int exercise04(int npart, int niters) {
    // CPU vars
    float *xin, *yin, *zin;
    float *xout, *yout, *zout;
    float *m;

    // GPU vars
    float *xin_dev, *yin_dev, *zin_dev;
    float *xout_dev, *yout_dev, *zout_dev;
    float *m_dev;

    // Exchange vars
    float *x_dev, *y_dev, *z_dev;

    // kernel parameters
    float dt = 0.2f;
    float val = 0.2f;
    float execution_time = 0.0f;

    // CPU memory allocations inputs
    xin = (float*)malloc(npart * sizeof(float));
    yin = (float*)malloc(npart * sizeof(float));
    zin = (float*)malloc(npart * sizeof(float));
    m = (float*)malloc(npart * sizeof(float));

    // CPU memory allocation  outputs
    xout = (float*)malloc(npart * sizeof(float));
    yout = (float*)malloc(npart * sizeof(float));
    zout = (float*)malloc(npart * sizeof(float));

    // Initialize particles with random values
    init(xin, npart);
    init(yin, npart);
    init(zin, npart);
    init(m, npart);

    // GPU memory allocation inputs
    checkCudaErrors(cudaMalloc((void **)&xin_dev, npart * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&yin_dev, npart * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&zin_dev, npart * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&m_dev, npart * sizeof(float)));

    // GPU memory allocation outputs
    checkCudaErrors(cudaMalloc((void **)&xout_dev, npart * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&yout_dev, npart * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&zout_dev, npart * sizeof(float)));

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

    // Debug
    /*
 *  printf("Sampling INPUT\n");
    Print(xin);
    */

    // START measure time
    cudaEventRecord(start, 0);

    // Copy data from CPU to GPU
    checkCudaErrors(cudaMemcpy(xin_dev, xin, npart * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(yin_dev, yin, npart * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(zin_dev, zin, npart * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(m_dev, m, npart * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel execution
    for (int i = 0; i < niters; i++) {
        kernel<<<blocks, threads>>>(
                        xin_dev, yin_dev, zin_dev,
                        xout_dev, yout_dev, zout_dev,
                        m_dev, npart, dt, val
        );
        cudaDeviceSynchronize();

        // Data exchange
        x_dev = xin_dev;
        y_dev = yin_dev;
        z_dev = zin_dev;
        xin_dev = xout_dev;
        yin_dev = yout_dev;
        zin_dev = zout_dev;
        xout_dev = x_dev;
        yout_dev = y_dev;
        zout_dev = z_dev;
    }

    // Copy data from GPU to CPU
    checkCudaErrors(cudaMemcpy(xout, xout_dev, npart * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(yout, yout_dev, npart * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(zout, zout_dev, npart * sizeof(float), cudaMemcpyDeviceToHost));

    // STOP measure time
    cudaEventRecord(stop, 0);

    // Debug
    /*
    printf("Sampling OUTPUT\n");
    Print(xout);
    */

    // Calculate time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&execution_time, start, stop);
    printf("kernel execution: %f seconds\n", (execution_time / 1000.0f));

    // Free GPU memory allocations
    checkCudaErrors(cudaFree(xin_dev));
    checkCudaErrors(cudaFree(yin_dev));
    checkCudaErrors(cudaFree(zin_dev));
    checkCudaErrors(cudaFree(xout_dev));
    checkCudaErrors(cudaFree(yout_dev));
    checkCudaErrors(cudaFree(zout_dev));
    checkCudaErrors(cudaFree(m_dev));

    // Free memory
    free(xin);
    free(yin);
    free(zin);
    free(xout);
    free(yout);
    free(zout);
    free(m);

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
    cout << "Executing ..." << endl;
    exercise04(npart, niters);

    return 0;
}

