#include <stdio.h>
#include <curand.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include "cublas_v2.h"

#define DEFAULT_M     3
#define DEFAULT_N     3
#define DEFAULT_K     2
#define DEFAULT_ALPHA 1
#define DEFAULT_BETA  1

using namespace std;

static const char help[] =
    "Usage: exercise05 [-m, -n, -k, -a, -b number] [-h]\n"
    "Description: This program computes the following expresion:\n"
    "              C = (alpha)C + (beta)A * B\n"
    "              Where: C(n, m), B(n, k), A(k, m) are matrices\n"
    "              and alpha, beta are scalars.\n"
    "  -m, -n, -k, Specify matrix dimensions.\n"
    "  -a, -b      Specify the alpha, beta scalars respectively.\n"
    "  -h          Prints this help message.\n";

void print_matrix(float *A, int m, int n, string label) {
    cout << label << " = np.array([" << endl;

    for(int i = 0; i < m; ++i) {
        cout << "\t[";
        for(int j = 0; j < n; ++j)
            cout << A[j * m + i] << ", ";
        cout << "]," << endl;
    }

    cout << "])" << endl;
}

void Print_input_as_python(float *A_dev, float *B_dev, float *C_dev, int m, int n, int k, int a, int b) {
    // CPU memory allocations
    float *A_host = (float *)malloc(n * k * sizeof(float));
    float *B_host = (float *)malloc(k * m * sizeof(float));
    float *C_host = (float *)malloc(n * m * sizeof(float));

    // Copy from GPU device to host
    cudaMemcpy(A_host, A_dev, n * k * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(B_host, B_dev, k * m * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(C_host, C_dev, n * m * sizeof(float),cudaMemcpyDeviceToHost);

    // Print matrix content as python format
    cout << "import numpy as np" << endl;
    cout << "alpha = " << a << endl;
    cout << "beta = " << b << endl;
    print_matrix(A_host, n, k, "A");
    print_matrix(B_host, k, m, "B");
    print_matrix(C_host, n, m, "C");

    // Free memory allocated in host
    free(A_host);
    free(B_host);
    free(C_host);
}

void Print_output_as_python(float *C_dev, int n, int m) {
    // CPU memory allocation
    float *C_host = (float *)malloc(n * m * sizeof(float));

    // Copy from GPU device to host
    cudaMemcpy(C_host, C_dev, n * m * sizeof(float),cudaMemcpyDeviceToHost);

    // Print result matrix content as python format
    print_matrix(C_host, n, m, "R");
    cout << "print('Python:________________________')" << endl;
    cout << "print((alpha * C) + (beta * (A.dot(B))))" << endl;
    cout << "print('CuBLAS:________________________')" << endl;
    cout << "print(R)" << endl;

    // Free memory allocated in host
    free(C_host);
}

void gemm_computation(float *A, float *B, float *C, int Cm, int Cn, int Ck, float a, float b) {
    // We must this exchange due to FORTRAN and C format differences:
    // C :      C(n,m) = B(n,k) * A(k,m)
    // Fortran: C(m,n) = A(m,k) * B(k,n)
    int m = Cn;
    int n = Cm;
    int k = Ck;

    // Leading dimensions
    int lda = m;
    int ldb = k;
    int ldc = m;

    // Scalars
    float *alpha = &a;
    float *beta = &b;
    float execution_time = 0.0f;

    // Structures to measure time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // START measure time
    cudaEventRecord(start, 0);

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // STOP measure time
    cudaEventRecord(stop, 0);

    // Calculate time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&execution_time, start, stop);
    printf("### GAMM execution: %f seconds\n", (execution_time / 1000.0f));

    // Destroy the handle
    cublasDestroy(handle);
}

void init_rand_matrix_GPU(float *A, int m, int n) {
    // Random number generator
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

    // Use clock as seed
    curandSetPseudoRandomGeneratorSeed(generator, clock());

    // Set values to matrix
    curandGenerateUniform(generator, A, n * m);
}

void matrices_computation(int m, int n, int k, float a, float b) {
    float *A_dev;
    float *B_dev;
    float *C_dev;

    // GPU memory allocations
    cudaMalloc(&A_dev, n * k * sizeof(float));
    cudaMalloc(&B_dev, k * m * sizeof(float));
    cudaMalloc(&C_dev, n * m * sizeof(float));

    // Init matrices A, B and C in GPU space
    init_rand_matrix_GPU(A_dev, n, k);
    init_rand_matrix_GPU(B_dev, k, m);
    init_rand_matrix_GPU(C_dev, n, m);

    // NOTE - PART 1: This part is just for testing
    //Print_input_as_python(A_dev, B_dev, C_dev, m, n, k, a, b);

    // Multiply A and B on GPU
    gemm_computation(A_dev, B_dev, C_dev, m, n, k, a, b);

    // NOTE - PART 2: This part is just for testing
    //Print_output_as_python(C_dev, n, m);

    // Free allocated memory on GPU
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
}

int main(int argc, char **argv) {
    int opt;

    // Matrix dimensions
    int m = DEFAULT_M;
    int n = DEFAULT_N;
    int k = DEFAULT_K;

    // Scalars
    float a = DEFAULT_ALPHA;
    float b = DEFAULT_BETA;

    while ((opt = getopt(argc, argv, "a:b:m:n:k:h")) != EOF) {
        switch (opt) {
            case 'a':
                a = atoi(optarg);
                break;
            case 'b':
                b = atoi(optarg);
                break;
            case 'm':
                m = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'k':
                k = atoi(optarg);
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

    matrices_computation(m, n, k, a, b);

    return 0;
}
