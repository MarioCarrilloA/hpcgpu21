#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <math.h>
#include <unistd.h>

#define N 32
#define DEFAULT_NUM_ITERATIONS 1

using namespace std;

static const char help[] =
    "Usage: exercise01 [-f number] [-n number] [-h]\n"
    "Description:\n"
    "  -f number:     Specifies the function to be executed from the\n"
    "                 4 available in exercise 1.\n"
    "  -n number:     Specifies how many computing iterations will be\n"
    "                 executed for the chosen function.\n"
    "  -h:            Prints this help message.\n";

template <class T>
void Print(T *A) {
    int i;

    for (i = 0; i < N; i++)
        cout << i <<  "\t"  << A[i] << endl;
}

template <class T2>
void init(T2 *A) {
    int i;

    for (i = 0; i < N; i++)
        A[i] = i + 1;
}


// GPU Kernel with floats
__global__ void foo1(float *A, int niters) {
    int tid=threadIdx.x;
    float val;

    val = A[tid];
    for (int iter=0; iter < niters; iter++) {
        val = (sqrtf(powf(val, 2)) + 5.0f) - 101.0f;
        val = (val / 3.0f) + 102.0f;
        val = (val + 1.07f) - 103.0f;
        val = (val / 1.037f) + 104.0f;
        val = (val + 3.00f) - 105.0f;
        val = (val / 0.22f) + 106.0f;
    }
    A[tid] = val;
}


// GPU Kernel with doubles
__global__ void foo2(double *A, int niters) {
    int tid=threadIdx.x;
    double val;

    val = A[tid];
    for (int iter=0; iter < niters; iter++) {
        val = (sqrt(pow(val, 2)) + 5.0) - 101.0;
        val = (val / 3.0) + 102.0;
        val = (val + 1.07) - 103.0;
        val = (val / 1.037) + 104.0;
        val = (val + 3.00) - 105.0;
        val = (val / 0.22) + 106.0;
    }
    A[tid] = val;
}

// CPU host with floats
void foo3(float *A, int niters) {
    float val;
    for (int tid = 0; tid < N; tid++) {
        val = A[tid];
        for (int iter=0; iter < niters; iter++) {
            val = (sqrtf(powf(val, 2)) + 5.0f) - 101.0f;
            val = (val / 3.0f) + 102.0f;
            val = (val + 1.07f) - 103.0f;
            val = (val / 1.037f) + 104.0f;
            val = (val + 3.00f) - 105.0f;
            val = (val / 0.22f) + 106.0f;
        }
        A[tid] = val;
    }
}


// CPU host with doubles
void foo4(double *A, int niters) {
    double val;
    for (int tid = 0; tid < N; tid++) {
        val = A[tid];
        for (int iter=0; iter < niters; iter++) {
            val = (sqrt(pow(val, 2)) + 5.0) - 101.0;
            val = (val / 3.0) + 102.0;
            val = (val + 1.07) - 103.0;
            val = (val / 1.037) + 104.0;
            val = (val + 3.00) - 105.0;
            val = (val / 0.22) + 106.0;
        }
        A[tid] = val;
    }
}


// GPU Kernel with floats / fabs instead of sqrt, pow
__global__ void foo5(float *A, int niters) {
    int tid=threadIdx.x;
    float val;

    val = A[tid];
    for (int iter=0; iter < niters; iter++) {
        val = (fabs(val) + 5.0f) - 101.0f;
        val = (val / 3.0f) + 102.0f;
        val = (val + 1.07f) - 103.0f;
        val = (val / 1.037f) + 104.0f;
        val = (val + 3.00f) - 105.0f;
        val = (val / 0.22f) + 106.0f;
    }
    A[tid] = val;
}

// GPU Kernel with doubles / fabs instead of sqrt, pow
__global__ void foo6(double *A, int niters) {
    int tid=threadIdx.x;
    double val;

    val = A[tid];
    for (int iter=0; iter < niters; iter++) {
        val = (fabs(val) + 5.0) - 101.0;
        val = (val / 3.0) + 102.0;
        val = (val + 1.07) - 103.0;
        val = (val / 1.037) + 104.0;
        val = (val + 3.00) - 105.0;
        val = (val / 0.22) + 106.0;
    }
    A[tid] = val;
}

int exercise01(int functid, int niters) {
    if (functid < 1 || functid > 6)
        return 1;

    switch(functid) {
        case 1:
            float *A1;
            float *dev_A1;
            A1 = (float *)malloc(sizeof(float) * N);
            init(A1);
            cudaMalloc((void **)&dev_A1, sizeof(float) * N);
            cudaMemcpy(dev_A1, A1, sizeof(float) * N, cudaMemcpyHostToDevice);
            foo1 <<< 1, N >>>(dev_A1, niters);
            cudaMemcpy(A1, dev_A1, sizeof(float) * N, cudaMemcpyDeviceToHost);
            Print(A1);
            cudaFree(dev_A1);
            free(A1);
            break;

        case 2:
            double *A2;
            double *dev_A2;
            A2 = (double *)malloc(sizeof(double) * N);
            init(A2);
            cudaMalloc((void **)&dev_A2, sizeof(double) * N);
            cudaMemcpy(dev_A2, A2, sizeof(double) * N, cudaMemcpyHostToDevice);
            foo2 <<< 1, N >>>(dev_A2, niters);
            cudaMemcpy(A2, dev_A2, sizeof(double) * N, cudaMemcpyDeviceToHost);
            Print(A2);
            cudaFree(dev_A2);
            free(A2);
            break;

        case 3:
            float *A3;
            A3 = (float *)malloc(sizeof(float) * N);
            init(A3);
            foo3(A3, niters);
            Print(A3);
            free(A3);
            break;

        case 4:
            double *A4;
            A4 = (double *)malloc(sizeof(double) * N);
            init(A4);
            foo4(A4, niters);
            Print(A4);
            free(A4);
            break;

        case 5:
            float *A5;
            float *dev_A5;
            A5 = (float *)malloc(sizeof(float) * N);
            init(A5);
            cudaMalloc((void **)&dev_A5, sizeof(float) * N);
            cudaMemcpy(dev_A5, A5, sizeof(float) * N, cudaMemcpyHostToDevice);
            foo5 <<< 1, N >>>(dev_A5, niters);
            cudaMemcpy(A5, dev_A5, sizeof(float) * N, cudaMemcpyDeviceToHost);
            Print(A5);
            cudaFree(dev_A5);
            free(A5);
            break;

        case 6:
            double *A6;
            double *dev_A6;
            A6 = (double *)malloc(sizeof(double) * N);
            init(A6);
            cudaMalloc((void **)&dev_A6, sizeof(double) * N);
            cudaMemcpy(dev_A6, A6, sizeof(double) * N, cudaMemcpyHostToDevice);
            foo6 <<< 1, N >>>(dev_A6, niters);
            cudaMemcpy(A6, dev_A6, sizeof(double) * N, cudaMemcpyDeviceToHost);
            Print(A6);
            cudaFree(dev_A6);
            free(A6);
            break;
    }
    return 0;
}


int main(int argc, char **argv) {
    int opt;
    int functid = 1;
    int niters = DEFAULT_NUM_ITERATIONS;

    while ((opt = getopt(argc, argv, "f:n:h")) != EOF) {
        switch (opt) {
            case 'f':
                functid = atoi(optarg);
                break;
            case 'n':
                niters = atoi(optarg);
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

    cout << "Function: " << functid << "  Iterations number: " << niters << endl;
    exercise01(functid, niters);

    return 0;
}

