#include <stdio.h>
#include <iostream>
#include <curand.h>
#include <mma.h>
#include <unistd.h>
#include <helper_cuda.h>
#include "cublas_v2.h"

#define DEFAULT_MATRIX_M     16
#define DEFAULT_MATRIX_N     16
#define DEFAULT_MATRIX_K     16
#define DEFAULT_ALPHA        2.0
#define DEFAULT_BETA         2.0

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


void print_fullp_matrix(float *A_dev, int m, int n, string label) {
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


// WMMA kernel
__global__ void wmma_example(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta) {
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

//int main(int argc, char* argv[]) {
void matrices_computation(int m, int n, int k, float a, float b) {
    float *a_fp32;
    float *b_fp32;
    half *a_fp16;
    half *b_fp16;

    float *c;
    float *c_wmma;

    //float *c_host_wmma;

    // Structures to measure time
    curandGenerator_t gen;
    cudaEvent_t startWMMA;
    cudaEvent_t stopWMMA;
    checkCudaErrors(cudaEventCreate(&startWMMA));
    checkCudaErrors(cudaEventCreate(&stopWMMA));

    // Use tensor cores
    checkCudaErrors(cudaMalloc((void**)&a_fp32, m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&b_fp32, k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&a_fp16, m * k * sizeof(half)));
    checkCudaErrors(cudaMalloc((void**)&b_fp16, k * n * sizeof(half)));
    checkCudaErrors(cudaMalloc((void**)&c, m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&c_wmma, m * n * sizeof(float)));
    //c_host_wmma = (float*)malloc(m * n * sizeof(float));

    // Fill matrix with random data
    checkCudaErrors(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
    checkCudaErrors(curandGenerateUniform(gen, a_fp32, m * k));
    checkCudaErrors(curandGenerateUniform(gen, b_fp32, k * n));

    // Convert Fp32 to Fp16 beacause curand does not support Fp16
    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (a_fp16, a_fp32, m * k);
    convertFp32ToFp16 <<< (k * n + 255) / 256, 256 >>> (b_fp16, b_fp32, k * n);

    checkCudaErrors(curandGenerateUniform(gen, c, m * n));
    checkCudaErrors(curandDestroyGenerator(gen));
    checkCudaErrors(cudaMemcpy(c_wmma, c, m * n * sizeof(float), cudaMemcpyDeviceToDevice));

    float alpha = 2.0f;
    float beta = 2.0f;

    // First: using WMMA
    dim3 gridDim;
    dim3 blockDim;

    // Set block / threads dimensions
    // blockDim.x must be a multple of warpSize
    // 128x4 means we have 16 warps and a block computes a 64x64 output tile
    blockDim.x = 128;
    blockDim.y = 4;
    gridDim.x = (m + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
    gridDim.y = (n + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

     // Print debug output matrices as python format
     printf("import numpy as np\n");
     printf("alpha=%f\n", alpha);
     printf("beta=%f\n", beta);
     print_fullp_matrix(a_fp32, m, k, "A");
     print_fullp_matrix(b_fp32, k, n, "B");
     print_fullp_matrix(c, m, n, "C");

    // Execute kernel
    checkCudaErrors(cudaEventRecord(startWMMA));
    wmma_example <<< gridDim, blockDim >>> (a_fp16, b_fp16, c_wmma, m, n, k, alpha, beta);
    checkCudaErrors(cudaEventRecord(stopWMMA));

    // Print debug output matrices as python format
    printf("print((alpha * C) + (beta * (A.dot(B))))\n");
    print_fullp_matrix(c_wmma, m, n, "D");
    printf("print(D)\n");

    // Measure time
    float wmmaTime;
    checkCudaErrors(cudaEventSynchronize(stopWMMA));
    checkCudaErrors(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    checkCudaErrors(cudaEventDestroy(startWMMA));
    checkCudaErrors(cudaEventDestroy(stopWMMA));

    // Free memory
    checkCudaErrors(cudaFree(a_fp32));
    checkCudaErrors(cudaFree(b_fp32));
    checkCudaErrors(cudaFree(a_fp16));
    checkCudaErrors(cudaFree(b_fp16));

    checkCudaErrors(cudaFree(c));
    checkCudaErrors(cudaFree(c_wmma));

    //free(c_host_wmma);
    checkCudaErrors(cudaDeviceReset());
}

int main(int argc, char **argv) {
    int opt;

    // Matrix dimensions
    int m = DEFAULT_MATRIX_M;
    int n = DEFAULT_MATRIX_N;
    int k = DEFAULT_MATRIX_K;

    // Scalars
    float a = DEFAULT_ALPHA;
    float b = DEFAULT_BETA;

    while ((opt = getopt(argc, argv, "a:b:m:n:k:h")) != EOF) {
        switch (opt) {
            case 'a':
                a = atof(optarg);
                break;
            case 'b':
                b = atof(optarg);
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
