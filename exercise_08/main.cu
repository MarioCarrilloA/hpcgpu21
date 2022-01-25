#include <stdio.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <random>
#include <sstream>
#include <mma.h>
#include <unistd.h>
#include <helper_cuda.h>
#include "cublas_v2.h"

#define DEFAULT_MATRIX_M 1024
#define DEFAULT_MATRIX_N 2536

using std::cout;
using std::endl;
using std::setprecision;
using namespace std;

constexpr int FLOAT_MIN = -1.0;
constexpr int FLOAT_MAX = 1.0;
const int RANDOM_NUM_PRECISION = 5;

static const char help[] =
    "Usage: exercise08 [-m, -n number] [-h]\n"
    "Description: This program computes the following expresion:\n"
    "            y^T = x^T + A\n"
    "            Where: yT(n), n^T(m) are vectors and A(m, n) is a matrix\n"
    "  -m, -n,   Specify matrix/vector dimensions.\n"
    "  -h          Prints this help message.\n";

void print_mmv(float *A_dev, int m, int n, string label) {
    // CPU memory allocations
    float *A_host = (float *)malloc(m * n * sizeof(float));

    // Copy from GPU device to host
    checkCudaErrors(cudaMemcpy(A_host, A_dev, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    cout << label << " = np.array([" << endl;
    for(int i = 0; i < m; ++i) {
        cout << "\t[";
        for(int j = 0; j < n; ++j)
            cout << A_host[j * m + i] << ", ";
        cout << "]," << endl;
    }
cout << "])" << endl;

    free(A_host);
}

// Vector/Matrix multiplication  kernel
__global__ void mm_kernel(float *A, float *x, float *y, int m, int n) {
    extern __shared__ float sharedArr[];
    float* xi = (float*)&sharedArr[0];
    float* psum = (float*)&xi[m];

    // partial sum per thread
    for (int i = threadIdx.x; i < m; i+=blockDim.x) {
        // load x to shared memory to prevent reload of x
        *((float*)(&xi[0]) + i) = *((float*)(&x[0]) + i);

        int g = i + (blockIdx.x * m);
        psum[threadIdx.x] = xi[i] * A[g];
    }

    // bringing thread groups together
    __syncthreads();
    int off = blockDim.x/2;

    // start with 1/2 #threads
    while (off > 31) {
        //printf("#INSIDE WHILE");
        if(threadIdx.x < off) {
            psum[threadIdx.x] += psum[threadIdx.x+off];
        }
        // 1/2 no of threads involved
        off/=2;

        __syncthreads();
    }

    // Finalizing the last warp - No loop, no overhead, no __syncthreads in single warp
    if(threadIdx.x < 16) {
        __syncwarp();
        psum[threadIdx.x] += psum[threadIdx.x + 16];
    }
    if(threadIdx.x < 8) {
        __syncwarp();
        psum[threadIdx.x] += psum[threadIdx.x + 8];
    }
    if(threadIdx.x < 4) {
        __syncwarp();
        psum[threadIdx.x] += psum[threadIdx.x + 4];
    }
    if(threadIdx.x < 2) {
        __syncwarp();
        psum[threadIdx.x] += psum[threadIdx.x + 2];
    }
    if(threadIdx.x < 1) {
        __syncwarp();
        psum[threadIdx.x] += psum[threadIdx.x + 1];
    }

    y[blockIdx.x] = psum[threadIdx.x];
}


void init(float *A, int m, int n) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<> distr(FLOAT_MIN, FLOAT_MAX);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            stringstream ss;
            ss << setprecision(RANDOM_NUM_PRECISION) << distr(eng);
            A[j * m + i] = stof(ss.str());
        }
    }
}

void vec_mtx_computation(int m, int n) {
    // Device memory vars
    float *A_dev;
    float *x_dev;
    float *y_dev;

    // Host memory vars
    float *A_host;
    float *x_host;
    float *y_host;

    // Structures to measure time
    cudaEvent_t start;
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    A_host = (float *)malloc(m * n * sizeof(float));
    x_host = (float *)malloc(m * sizeof(float));
    y_host = (float *)malloc(n * sizeof(float));

    // Fill with [-1, 1] values
    init(A_host, m, n);
    init(x_host, 1, m);
    init(y_host, 1, n);

    // Use tensor cores
    checkCudaErrors(cudaMalloc((void**)&A_dev, m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&x_dev, m *  sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&y_dev, n * sizeof(float)));

    // Host memory handling
    checkCudaErrors(cudaMemcpy(A_dev, A_host, m * n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(x_dev, x_host, m * sizeof(float), cudaMemcpyHostToDevice));

    // Set number of threads/blocks
    dim3 blocks(n, 1, 1);
    dim3 threads(m, 1, 1);

    checkCudaErrors(cudaEventRecord(start));
    mm_kernel<<<blocks, threads, (m + m) * sizeof(float)>>>(A_dev, x_dev, y_dev, m, n);
    checkCudaErrors(cudaEventRecord(stop));

    // Measure time
    float execution_time;
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&execution_time, start, stop));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // *** UNCOMMENT THE LINE BLOCK BELOW TO PRINT RESULTS ***
    // This section print debug output matrices as python format.
    // So we can use the output of sbatch as python script.
    // -----------------------------------------------------------
    /*
    printf("import numpy as np\n");
    print_mmv(A_dev, m, n, "A");
    print_mmv(x_dev, 1, m, "x");
    printf("print('Python:________________________')\n");
    printf("print(x.dot(A))\n");
    print_mmv(y_dev, 1, n, "y");
    printf("print('CUDA MULT:________________________')\n");
    printf("print(y)\n");
    */

    // Print execution time
    printf("### WAMM execution: %f milliseconds\n", (execution_time));

    // Free memory
    checkCudaErrors(cudaFree(A_dev));
    checkCudaErrors(cudaFree(x_dev));
    checkCudaErrors(cudaFree(y_dev));
    free(A_host);
    free(x_host);
    free(y_host);
}

int main(int argc, char **argv) {
    int opt;

    // Matrix <6 vector dimensions
    int m = DEFAULT_MATRIX_M;
    int n = DEFAULT_MATRIX_N;

    while ((opt = getopt(argc, argv, "m:n:h")) != EOF) {
        switch (opt) {
            case 'm':
                m = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
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

    vec_mtx_computation(m, n);

    return 0;
}
