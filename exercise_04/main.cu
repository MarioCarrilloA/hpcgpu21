#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <math.h>
#include <unistd.h>
#include <helper_cuda.h>
#include <sys/time.h>

#define DEFAULT_NUM_ITERATIONS 10
#define DEFAULT_KERNEL_ID      1
#define DEFAULT_NUM_PARTICLES  9984
#define DEFAULT_NUM_TO_SHOW    10

using namespace std;

struct p {
    float x;
    float y;
    float z;
    float m;
};

static const char help[] =
    "Usage: exercise01 [-k number] [-i number] [-p number] [-h]\n"
    "Description:\n"
    "  -k number:     Specifies the kernel to be executed from the\n"
    "                 2 available in exercise 4.\n"
    "  -i number:     Specifies how many computing iterations will be\n"
    "                 executed for the chosen function.\n"
    "  -p number:     Number of particles to be processed\n"
    "  -h             Prints this help message.\n";

void Print(p *x) {
    for (int i = 0; i < DEFAULT_NUM_TO_SHOW; i++)
        cout << x[i].x << endl;
}

void init(p *xin, long npart) {
    for (int i = 0; i < npart; i++) {
        xin[i].x = (float(rand())/float((RAND_MAX)) * 10.0f) + 0.1f;
        xin[i].y = (float(rand())/float((RAND_MAX)) * 10.0f) + 0.1f;
        xin[i].z = (float(rand())/float((RAND_MAX)) * 10.0f) + 0.1f;
        xin[i].m = (float(rand())/float((RAND_MAX)) * 10.0f) + 0.1f;
    }
}

// GPU Kernel 1
__global__ void kernel_1(p *xin, p *xout, long int npart, double dt, double val) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int maxrad = 1.0;
    float f, dsq;

    while (i < npart) {
        xout[i].x = xin[i].x;
        xout[i].y = xin[i].y;
        xout[i].z = xin[i].z;
        f = 0.0f;
        dsq = 0.0f;
        for (int j = 0; j < npart; j++) {
            dsq = (
                    powf(xin[i].x - xin[j].x, 2.0f) +
                    powf(xin[i].y - xin[j].y, 2.0f) +
                    powf(xin[i].x - xin[j].x, 2.0f)
            );

            if (dsq < maxrad && dsq != 0 && i != j) {
                f += xin[i].m * xin[j].m * (xin[i].x - xin[j].x) / dsq;
            }
        }
        double s = f * dt * val;
        xout[i].x += s;
        xout[i].y += s;
        xout[i].z += s;

        i+=off;
    }
}

// GPU Kernel 2
__global__ void kernel_2(p *xin, p *xout, long int npart, double dt, double val) {
    //int tid=threadIdx.x;
}


void execute_kernel(p *xin, p *xout, int npart, int niters, int kernelid) {
    p *x_dev;
    p *xin_dev;
    p *xout_dev;
    float dt = 0.5f;
    float val = 0.5f;
    float execution_time = 0.0f;

    // Structures to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Set number of threads/blocks
    dim3 block(64, 64, 1);
    dim3 threads(32, 32, 1);

    checkCudaErrors(cudaMalloc((void **)&xin_dev, sizeof(p) * npart));
    checkCudaErrors(cudaMalloc((void **)&xout_dev, sizeof(p) * npart));
    checkCudaErrors(cudaMemcpy(xin_dev, xin, sizeof(p) * npart, cudaMemcpyHostToDevice));

    // START measure time
    cudaEventRecord(start, 0);

    if (kernelid == 1) {
        // Kernel 1 execution
        for (int i = 0; i < niters; i++) {
            kernel_1<<<block, threads>>>(xin_dev, xout_dev, npart, dt, val);

            // Exchange pointers
            x_dev = xin_dev;
            xin_dev = xout_dev;
            xout_dev = x_dev;
        }

    } else {
        // Kernel 2 execution
        for (int i = 0; i < niters; i++) {
            kernel_2<<<block, threads>>>(xin_dev, xout_dev, npart, dt, val);

            // Exchange pointers
            x_dev = xin_dev;
            xin_dev = xout_dev;
            xout_dev = x_dev;
        }
    }

    // STOP measure time
    cudaEventRecord(stop, 0);

    // Calculate time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&execution_time, start, stop);
    //checkCudaErrors(cudaMemcpy(xout, xout_dev, sizeof(p) * npart, cudaMemcpyDeviceToHost));
    //Print(xout);
    checkCudaErrors(cudaFree(xin_dev));
    checkCudaErrors(cudaFree(xout_dev));
    printf("kernel execution: %f seconds\n", (execution_time / 1000.0f));
}


int exercise04(int kernelid, int npart, int niters) {
    if (kernelid < 1 || kernelid > 2)
        return 1;

    p *xin = (p *)malloc(sizeof(p) * npart);
    p *xout = (p *)malloc(sizeof(p) * npart);
    init(xin, npart);
    execute_kernel(xin, xout, npart, niters, kernelid);
    free(xin);
    free(xout);

    return 0;
}


int main(int argc, char **argv) {
    int opt;
    int kernelid = DEFAULT_KERNEL_ID;
    int niters = DEFAULT_NUM_ITERATIONS;
    long int npart = DEFAULT_NUM_PARTICLES;

    while ((opt = getopt(argc, argv, "k:i:p:h")) != EOF) {
        switch (opt) {
            case 'k':
                kernelid = atoi(optarg);
                break;
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

    cout << "Kernel: " << kernelid << endl;
    cout << "Particles: " << npart << endl;
    cout << "Iterations: " << niters << endl;
    cout << "Executing ..." << endl;
    exercise04(kernelid, npart, niters);

    return 0;
}

