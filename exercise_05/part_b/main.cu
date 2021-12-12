#include <stdio.h>
#include <curand.h>
#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <helper_cuda.h>
#include "cublas_v2.h"
#include <mma.h>

#define DEFAULT_M     3
#define DEFAULT_N     3
#define DEFAULT_K     2
#define DEFAULT_ALPHA 1
#define DEFAULT_BETA  1

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

using namespace std;
using namespace nvcuda;

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

void init_rand_matrix_GPU(float *A, int m, int n) {
    // Random number generator
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);

    // Use clock as seed
    curandSetPseudoRandomGeneratorSeed(generator, clock());

    // Set values to matrix
    curandGenerateUniform(generator, A, n * m);
}

__global__ void wmma_computation(half *A, half *B, float *C, int Cm, int Cn, int Ck, float alpha, float beta) {
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

    // Tile using a 2D grid
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);

    // Define the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> A_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> B_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> BA_frag;
    wmma::fill_fragment(BA_frag, 0.0f);

    // Loop for the computation
    for(int i = 0; i < k; i += WMMA_K){
        int rowA = warpN * WMMA_N;
        int colA = i;
        int rowB = i;
        int colB = warpM * WMMA_M;

        if(rowA < k && colA < m && rowB < n && colB < k){
            // Load the inputs
            wmma::load_matrix_sync(A_frag, A + rowA * lda + colA, lda);
            wmma::load_matrix_sync(B_frag, B + rowB * ldb + colB, ldb);

            // Matrix multiplication
            wmma::mma_sync(BA_frag, A_frag, B_frag, BA_frag);
        }
    }

    // C = alpha * C + beta * B * A
    int rowC = warpN * WMMA_N;
    int colC = warpM * WMMA_M;

    if(rowC < n && colC < m){
        wmma::load_matrix_sync(C_frag, C + rowC * ldc + colC, ldc, wmma::mem_row_major);

        for(int i = 0; i < C_frag.num_elements; i++){
            C_frag.x[i] = alpha * C_frag.x[i] + beta * BA_frag.x[i];
        }

        wmma::store_matrix_sync(C + rowC * ldc + colC, C_frag, ldc, wmma::mem_row_major);
    }

}

__global__ void castF32ToF16 (half *f16, float *f32, int n){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n){
        f16[i] = f32[i];
    }
}

void matrices_computation(int m, int n, int k, float alpha, float beta) {
    float *A_f32;
    float *B_f32;
    float *C_f32;

    half *A_f16;
    half *B_f16;

    // Scalars
    float execution_time = 0.0f;

    // GPU memory allocations
    checkCudaErrors(cudaMalloc(&A_f32, n * k * sizeof(float)));
    checkCudaErrors(cudaMalloc(&B_f32, k * m * sizeof(float)));
    checkCudaErrors(cudaMalloc(&C_f32, n * m * sizeof(float)));

    checkCudaErrors(cudaMalloc(&A_f16, n * k * sizeof(half)));
    checkCudaErrors(cudaMalloc(&B_f16, k * m * sizeof(half)));

    // Init matrices A, B and C in GPU space
    init_rand_matrix_GPU(A_f32, n, k);
    init_rand_matrix_GPU(B_f32, k, m);
    init_rand_matrix_GPU(C_f32, n, m);

    // Cast A, B to fp16 half type
    castF32ToF16 <<<(n * k + 255) / 256, 256 >>> (A_f16, A_f32, n*k);
    castF32ToF16 <<<(k * m + 255) / 256, 256 >>> (B_f16, B_f32, k*m);

    // NOTE - PART 1: This part is just for testing
    //Print_input_as_python(A_dev, B_dev, C_dev, m, n, k, a, b);

    // Multiply A and B on GPU
    dim3 gridDim;
    dim3 blockDim;
    blockDim.x = 128;
    blockDim.y = 4;
    gridDim.x = (n + (WMMA_N * blockDim.x / 32 - 1)) / (WMMA_N * blockDim.x / 32);
    gridDim.y = (m + m * blockDim.y - 1) / (WMMA_N * blockDim.y);

    // Structures to measure time
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // START measure time
    checkCudaErrors(cudaEventRecord(start, 0));

    // Call the kernel for the matrix computation
    wmma_computation <<< gridDim, blockDim >>> (A_f16, B_f16, C_f32, m, n, k, alpha, beta);

    // STOP measure time
    checkCudaErrors(cudaEventRecord(stop, 0));

    // Calculate time
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&execution_time, start, stop));
    printf("### GAMM execution: %f seconds\n", (execution_time / 1000.0f));

    // Destroy the handle
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // NOTE - PART 2: This part is just for testing
    //Print_output_as_python(C_dev, n, m);

    // Free allocated memory on GPU
    checkCudaErrors(cudaFree(A_f32));
    checkCudaErrors(cudaFree(B_f32));
    checkCudaErrors(cudaFree(C_f32));
    checkCudaErrors(cudaFree(A_f16));
    checkCudaErrors(cudaFree(B_f16));
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
