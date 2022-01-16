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

#define DEFAULT_MATRIX_M     16
#define DEFAULT_MATRIX_N     16
#define DEFAULT_MATRIX_K     16
#define DEFAULT_ALPHA        3.0
#define DEFAULT_BETA         1.0

using std::cout;
using std::endl;
using std::setprecision;

constexpr int FLOAT_MIN = -1.0;
constexpr int FLOAT_MAX = 1.0;

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
const int RANDOM_NUM_PRECISION = 5;

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
__global__ void wmma_example(half *A, half *B, half *C, float *Cfp, int M, int N, int K, float alpha, float beta) {
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;
    int ldt = N;

    // Tile with a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    // step through A and B
    for (int i = 0; i < K; i += WMMA_K) {
        int a_row = warpM * WMMA_M;
        int a_col = i;

        int b_row = i;
        int b_col = warpN * WMMA_N;

        // Bounds checking
        if (a_row < M && a_col < K && b_row < K && b_col < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + a_row + a_col * lda, lda);
            wmma::load_matrix_sync(b_frag, B + b_row + b_col * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // C multiplication
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> cm_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> ct_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cmacc_frag;
    wmma::fill_fragment(cmacc_frag, 0.0f);

    // Step through C and C^T
    for (int i = 0; i < N; i+= WMMA_N) {
        int c_row = warpM * WMMA_M;
        int c_col = i;

        int ct_row = i;
        int ct_col = warpM * WMMA_M;

        // Bounds checking
        if (c_row < M && c_col < N && ct_row < N && ct_col < M) {
            // Load the inputs
            //int aaa = 0;
            wmma::load_matrix_sync(cm_frag, C + c_row + c_col * ldc, ldc);
            wmma::load_matrix_sync(ct_frag, C + ct_row + ct_col * ldt, ldt);

            // Perform the matrix multiplication
            wmma::mma_sync(cmacc_frag, cm_frag, ct_frag, cmacc_frag);
            wmma::store_matrix_sync(Cfp + c_row + c_col * ldc, cmacc_frag, ldc, wmma::mem_col_major);
        }
    }


    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int c_row = warpM * WMMA_M;
    int c_col = warpN * WMMA_N;

    if (c_row < M && c_col < N) {
        wmma::load_matrix_sync(c_frag, Cfp + c_row + c_col * ldc, ldc, wmma::mem_col_major);

        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output in C full precision matrix
        wmma::store_matrix_sync(Cfp + c_row + c_col * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}


void init(float *A, int m, int n) {
    std::random_device rd;
    std::default_random_engine eng(rd());
    //std::uniform_real_distribution<> distr(-1.0, 1.0);
    std::uniform_real_distribution<> distr(FLOAT_MIN, FLOAT_MAX);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            stringstream ss;
            ss << setprecision(RANDOM_NUM_PRECISION) << distr(eng);
            A[j * m + i] = stof(ss.str());
        }
    }
}

void matrices_computation(int m, int n, int k, float a, float b) {
    // Device memory vars
    float *A_fp32;
    float *B_fp32;
    float *C_temp;
    float *C_wmma;

    // Host memory vars
    float *A_host;
    float *B_host;
    float *C_host;

    // Alpha and Beta
    float alpha = a;
    float beta = b;

    // Half precision vars
    half *A_fp16;
    half *B_fp16;
    half *C_fp16;

    // Structures to measure time
    cudaEvent_t startWMMA;
    cudaEvent_t stopWMMA;
    checkCudaErrors(cudaEventCreate(&startWMMA));
    checkCudaErrors(cudaEventCreate(&stopWMMA));

    A_host = (float *)malloc(m * k * sizeof(float));
    B_host = (float *)malloc(k * n * sizeof(float));
    C_host = (float *)malloc(m * n * sizeof(float));

    // Fill with [-1, 1] values
    init(A_host, m, k);
    init(B_host, k, n);
    init(C_host, m, n);

    // Use tensor cores
    checkCudaErrors(cudaMalloc((void**)&A_fp32, m * k * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&B_fp32, k * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&C_temp, m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&C_wmma, m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&A_fp16, m * k * sizeof(half)));
    checkCudaErrors(cudaMalloc((void**)&B_fp16, k * n * sizeof(half)));
    checkCudaErrors(cudaMalloc((void**)&C_fp16, m * n * sizeof(half)));

    // Host memory handling
    checkCudaErrors(cudaMemcpy(A_fp32, A_host, m * k * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B_fp32, B_host, k * n * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C_temp, C_host, m * n * sizeof(float), cudaMemcpyHostToDevice));
    free(A_host);
    free(B_host);
    free(C_host);

    // Convert Fp32 to Fp16 beacause curand does not support Fp16
    convertFp32ToFp16 <<< (m * k + 255) / 256, 256 >>> (A_fp16, A_fp32, m * k);
    convertFp32ToFp16 <<< (k * n + 255) / 256, 256 >>> (B_fp16, B_fp32, k * n);

    // Fill C matriix
    checkCudaErrors(cudaMemcpy(C_wmma, C_temp, m * n * sizeof(float), cudaMemcpyDeviceToDevice));
    convertFp32ToFp16 <<< (m * n + 255) / 256, 256 >>> (C_fp16, C_wmma, m * n);

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
     print_fullp_matrix(A_fp32, m, k, "A");
     print_fullp_matrix(B_fp32, k, n, "B");
     print_fullp_matrix(C_temp, m, n, "C");

    // Execute kernel
    checkCudaErrors(cudaEventRecord(startWMMA));
    wmma_example <<< gridDim, blockDim >>> (A_fp16, B_fp16, C_fp16, C_wmma, m, n, k, alpha, beta);
    checkCudaErrors(cudaEventRecord(stopWMMA));

    // Print debug output matrices as python format
    printf("print('Python:________________________')\n");
    printf("print((beta * C.dot(C.T)) + (alpha * (A.dot(B))))\n");
    print_fullp_matrix(C_wmma, m, n, "D");
    printf("print('WMMA  :________________________')\n");
    printf("print(D)\n");

    // Measure time
    float wmmaTime;
    checkCudaErrors(cudaEventSynchronize(stopWMMA));
    checkCudaErrors(cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA));
    checkCudaErrors(cudaEventDestroy(startWMMA));
    checkCudaErrors(cudaEventDestroy(stopWMMA));

    // Free memory
    checkCudaErrors(cudaFree(A_fp32));
    checkCudaErrors(cudaFree(B_fp32));
    checkCudaErrors(cudaFree(A_fp16));
    checkCudaErrors(cudaFree(B_fp16));

    checkCudaErrors(cudaFree(C_temp));
    checkCudaErrors(cudaFree(C_wmma));

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
