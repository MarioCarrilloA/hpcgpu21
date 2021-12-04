#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <math.h>
#include <unistd.h>
#include <helper_cuda.h>
#include <sys/time.h>

#define DEFAULT_NUM_ITERATIONS 1000
#define DEFAULT_KERNEL_ID      1
#define DEFAULT_NUM_PARTICLES  1000000
#define DEFAULT_NUM_TO_SHOW    10

using namespace std;

struct p {
    double x;
    double y;
    double z;
    double mass;
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
        xin[i].x = (double(rand())/double((RAND_MAX)) * 10.0)+ 0.00001;
        xin[i].y = (double(rand())/double((RAND_MAX)) * 10.0)+ 0.00001;
        xin[i].z = (double(rand())/double((RAND_MAX)) * 10.0)+ 0.00001;
        xin[i].mass = (double(rand())/double((RAND_MAX)) * 10.0)+ 0.00001;
    }
}

// GPU Kernel 1
__global__ void kernel1(p *xin, p *xout, long int npart, double dt, double val) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int maxrad = 1.0;
    double f, dsq;

    while (i < npart) {
        xout[i].x = xin[i].x;
        xout[i].y = xin[i].y;
        xout[i].z = xin[i].z;
        f = 0.0;
        dsq = 0.0;
        for (int j = 0; j < npart; j++) {
            dsq = (
                    pow(xin[i].x - xin[j].x, 2.0) +
                    pow(xin[i].y - xin[j].y, 2.0) +
                    pow(xin[i].x - xin[j].x, 2.0)
            );

            if (dsq < maxrad && dsq != 0 && i != j) {
                f += xin[i].mass * xin[j].mass * (xin[i].x - xin[j].x) / dsq;
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
__global__ void kernel2(double *A, int niters) {
    //int tid=threadIdx.x;
}


void execute_k1(p *xin, p *xout, int npart, int niters) {
    p *x_dev;
    p *xin_dev;
    p *xout_dev;
    double dt = 0.5;
    double val = 0.5;
    struct timeval start, end;

    // Set number of threads/blocks
    dim3 block(64, 64, 1);
    dim3 threads(32, 32, 1);

    checkCudaErrors(cudaMalloc((void **)&xin_dev, sizeof(p) * npart));
    checkCudaErrors(cudaMalloc((void **)&xout_dev, sizeof(p) * npart));
    checkCudaErrors(cudaMemcpy(xin_dev, xin, sizeof(p) * npart, cudaMemcpyHostToDevice));
    gettimeofday(&start, 0);

    // Kernel execution
    for (int i = 0; i < niters; i++) {
        kernel1<<<block, threads>>>(xin_dev, xout_dev, npart, dt, val);

        // Exchange pointers
        x_dev = xin_dev;
        xin_dev = xout_dev;
        xout_dev = x_dev;
    }

    gettimeofday(&end, 0);
    checkCudaErrors(cudaMemcpy(xout, xout_dev, sizeof(p) * npart, cudaMemcpyDeviceToHost));
    Print(xout);
    checkCudaErrors(cudaFree(xin_dev));
    checkCudaErrors(cudaFree(xout_dev));
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double execution_time = seconds + microseconds * 1e-6;
    printf("kernel execution: %.3f seconds\n", execution_time);
}


int exercise04(int kernelid, int npart, int niters) {
    if (kernelid < 1 || kernelid > 2)
        return 1;

    p *xin = (p *)malloc(sizeof(p) * npart);
    p *xout = (p *)malloc(sizeof(p) * npart);
    init(xin, npart);
    //Print(xin);

    if (kernelid == 1) {
        execute_k1(xin, xout, npart, niters);

    } else if (kernelid == 2) {
        printf("Execute k2\n");
    } else {
        free(xin);
        free(xout);
        printf("error: Unknow kernel\n");
        cerr << help << endl;
        return 1;
    }

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

