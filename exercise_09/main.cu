#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <math.h>
#include <unistd.h>
#include <helper_cuda.h>
#include <sys/time.h>

// Local headers
#include <kernels.h>

#define DEFAULT_NUM_ITERATIONS 1000
#define DEFAULT_NUM_PARTICLES  80000
#define DEFAULT_NUM_TO_SHOW    10
#define MAX_THREADS_PER_BLOCK 1024

// Exercise 09 consists of 2 tasks. Therefore this number
// determines which one to execute
#define EXERCISE_TASK_NUM 2

using namespace std;

struct results {
    float E;
    float val;
    float dt;
};

static const char help[] =
    "Usage: exercise09 [-k number] [-i number] [-p number] [-h]\n"
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

float max_mass(p x, long npart) {
    float maxm = 0.0;
    for (int i = 0; i < npart; i++) {
        if (x.m[i] > maxm) {
            maxm = x.m[i];
        }
    }

    return maxm;
}

int execute_kernel(p xin, p xout, int npart, int niters) {
    p x_dev;
    p xin_dev;
    p xout_dev;
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

    // GPU memory allocations/transfers
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

    // #########################################################################
    // Stuff for all kernels
    // #########################################################################
    float *F;
    float *E;
    float *dt;
    float *val;

    // GPU vars
    float *F_dev;
    float *E_dev;
    float *dt_dev;
    float *val_dev;

    F = (float*)malloc(sizeof(float));
    E = (float*)malloc(sizeof(float));
    dt = (float*)malloc(sizeof(float));
    val = (float*)malloc(sizeof(float));
    *F = 0.0f;
    *E = 0.0f;
    *dt = 0.5f;
    *val = 0.5f;

    // Find max mass in particles
    float maxm = max_mass(xin, npart);
    float M = maxm * 1000.0f;

    // CPU results array
    results r[niters];
    // #########################################################################


    // The Exercise 09 is composed of 2 tasks, then the
    // kernels will be executed  according to task number.
    if (EXERCISE_TASK_NUM == 1) {
        // GPU memory allocations
        checkCudaErrors(cudaMalloc((void **)&F_dev, sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&E_dev, sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&dt_dev, sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&val_dev, sizeof(float)));
        checkCudaErrors(cudaMemcpy(F_dev, F, sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(E_dev, E, sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(dt_dev, dt, sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(val_dev, val, sizeof(float), cudaMemcpyHostToDevice));

        // START measure time
        cudaEventRecord(start, 0);

        // Kernel 2 - execution, calculate F
        kernel2<<<blocks, threads, 1024 * sizeof(float)>>>(xin_dev, F_dev, npart, M, dt_dev, val_dev);
        cudaDeviceSynchronize();

        // Execute kernel sequence
        for (int i = 0; i < niters; i++) {
            // Kernel 1 - execution, exercise 04/07
            kernel1<<<blocks, threads, sizeof(float) * 1024 * 12>>>(xin_dev, xout_dev, npart, dt_dev, val_dev);
            cudaDeviceSynchronize();

            // Kernel 2 - execution, calculate F
            kernel2<<<blocks, threads, 1024 * sizeof(float)>>>(xin_dev, F_dev, npart, M, dt_dev, val_dev);
            cudaDeviceSynchronize();

            // Exchange pointers
            x_dev = xin_dev;
            xin_dev = xout_dev;
            xout_dev = x_dev;

            // Kernel 3 - execution  (NOW  / OLD), calculate E
            kernel3<<<blocks, threads, 1024 * sizeof(float)>>>(xout_dev, xin_dev, E_dev, npart);
            cudaDeviceSynchronize();

            // Store values
            checkCudaErrors(cudaMemcpy(E, E_dev, sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(dt, dt_dev, sizeof(float), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(val, val_dev, sizeof(float), cudaMemcpyDeviceToHost));
            r[i].E = *E;
            r[i].dt = *dt;
            r[i].val = *val;
        }

        // STOP measure time
        cudaEventRecord(stop, 0);

        // This just to hide a warning
        *dt = r[niters - 1].dt;


    } else if (EXERCISE_TASK_NUM == 2) {
        printf("Task 2\n");
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        if(prop.deviceOverlap) {
            printf("device porvides overlapping streams!!!\n");
        } else {
            printf("error: device provides NO overlapping streams\n");
            return 0;
        }

        cudaStream_t stream1;
        cudaStream_t stream2;
        cudaStream_t stream3;
        cudaStream_t stream4;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        cudaStreamCreate(&stream3);
        cudaStreamCreate(&stream4);

        // GPU memory allocations
        checkCudaErrors(cudaMalloc((void **)&F_dev, sizeof(float)));
        checkCudaErrors(cudaMemcpy(F_dev, F, sizeof(float), cudaMemcpyHostToDevice));

        // Asynchronous memory
        checkCudaErrors(cudaMallocHost((void **)&E_dev, sizeof(float)));
        checkCudaErrors(cudaMallocHost((void **)&dt_dev, sizeof(float)));
        checkCudaErrors(cudaMallocHost((void **)&val_dev, sizeof(float)));
        checkCudaErrors(cudaMemcpyAsync(E_dev, E, sizeof(float), cudaMemcpyHostToDevice, stream4));
        checkCudaErrors(cudaMemcpyAsync(dt_dev, dt, sizeof(float), cudaMemcpyHostToDevice, stream4));
        checkCudaErrors(cudaMemcpyAsync(val_dev, val, sizeof(float), cudaMemcpyHostToDevice, stream4));
        cudaStreamSynchronize(stream4);

        // START measure time
        cudaEventRecord(start, 0);

        // Kernel 2 - execution, calculate F
        kernel2<<<blocks, threads, 1024 * sizeof(float)>>>(xin_dev, F_dev, npart, M, dt_dev, val_dev);
        cudaDeviceSynchronize();

        // Execute kernel sequence
        for (int i = 0; i < niters; i++) {
            // Kernel 1 - execution, exercise 04/07
            kernel1<<<blocks, threads, sizeof(float) * 1024 * 12, stream1>>>(xin_dev, xout_dev, npart, dt_dev, val_dev);
            cudaDeviceSynchronize();

            // Kernel 2 - execution, calculate F
            kernel2<<<blocks, threads, 1024 * sizeof(float), stream2>>>(xin_dev, F_dev, npart, M, dt_dev, val_dev);
            cudaDeviceSynchronize();

            // Exchange pointers
            x_dev = xin_dev;
            xin_dev = xout_dev;
            xout_dev = x_dev;

            // Kernel 3 - execution  (NOW  / OLD), calculate E
            kernel3<<<blocks, threads, 1024 * sizeof(float), stream3>>>(xout_dev, xin_dev, E_dev, npart);
            cudaDeviceSynchronize();

            // Store values
            checkCudaErrors(cudaMemcpyAsync(E, E_dev, sizeof(float), cudaMemcpyDeviceToHost, stream4));
            checkCudaErrors(cudaMemcpyAsync(dt, dt_dev, sizeof(float), cudaMemcpyDeviceToHost, stream4));
            checkCudaErrors(cudaMemcpyAsync(val, val_dev, sizeof(float), cudaMemcpyDeviceToHost, stream4));
            cudaStreamSynchronize(stream4);

            r[i].E = *E;
            r[i].dt = *dt;
            r[i].val = *val;
        }

        // STOP measure time
        cudaEventRecord(stop, 0);
        cudaFreeHost(E);
        cudaFreeHost(dt);
        cudaFreeHost(val);

    } else {
        printf("error: Invalid task number\n");
    }


    // Copy data from GPU to CPU to show results
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

    //Print(xout);

    // Calculate time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&execution_time, start, stop);
    printf("kernel execution: %f seconds\n", (execution_time / 1000.0f));
    return 0;
}

int exercise09(int npart, int niters) {
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
    exercise09(npart, niters);

    return 0;
}

