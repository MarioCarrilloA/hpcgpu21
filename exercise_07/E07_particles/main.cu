#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <math.h>
#include <unistd.h>
#include <helper_cuda.h>
#include <sys/time.h>

#define DEFAULT_NUM_ITERATIONS 1000
#define DEFAULT_NUM_PARTICLES  40000
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
    "  -i number:     Specifies how many times the kernel will be\n"
    "                 executed.\n"
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

// GPU Kernel
__global__ void kernel(p *xin, p *xout, long int npart, double dt, double val) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int maxrad = 1.0;
    float f, dsq;
    float distx, disty, distz;
    int size = blockDim.x;

    extern __shared__ p x_shared[];
    p *xi = &x_shared[0];
    p *xj = &x_shared[blockDim.x];

    *((float4*)(&xi[0]) + t) = *((float4*)(&xin[g * blockDim.x]) + t);
        
    for(int ja = 0; ja < npart; ja+= size) {
        for(int jl = 0; jl < size; jl += blockDim.x){
            *((float4*)(&xj[jl]) + t) = *((float4*)(&xin[ja + jl]) + t);
        }
        __syncthreads();

        for(int j = ja; j < ja + size; j++){
            distx = xi[t].x - xj[j - ja].x;
            disty = xi[t].y - xj[j - ja].y;
            distz = xi[t].z - xj[j - ja].z;

            dsq = distx * distx + disty * disty + distz * distz;

            if (dsq < maxrad && dsq != 0 && i != j) {
                f += xi[t].m * xj[j - ja].m * (xi[t].x - xj[j - ja].x) / dsq;
            }
        }
    }
    double s = f * dt * val;
    xout[i].x = xi[t].x + s;
    xout[i].y = xi[t].y + s;
    xout[i].z = xi[t].z + s;

    /*
    while (i < npart) {
        
        

        i+=off;
    }*/

    /*
    while (i < npart) {
        xout[i].x = xin[i].x;
        xout[i].y = xin[i].y;
        xout[i].z = xin[i].z;
        f = 0.0f;
        dsq = 0.0f;
        distx = 0.0f;
        disty = 0.0f;
        distz = 0.0f;
        for (int j = 0; j < npart; j++) {
            distx = xin[i].x - xin[j].x;
            disty = xin[i].y - xin[j].y;
            distz = xin[i].z - xin[j].z;

            dsq = distx * distx + disty * disty + distz * distz;

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
    */
}


void execute_kernel(p *xin, p *xout, int npart, int niters) {
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
    dim3 block(1024, 1, 1);
    dim3 threads(1024, 1, 1);

     // START measure time
    cudaEventRecord(start, 0);

    // Memory management
    checkCudaErrors(cudaMalloc((void **)&xin_dev, sizeof(p) * npart));
    checkCudaErrors(cudaMalloc((void **)&xout_dev, sizeof(p) * npart));
    checkCudaErrors(cudaMemcpy(xin_dev, xin, sizeof(p) * npart, cudaMemcpyHostToDevice));

    // for dynamic allocation of shared memory
    // Kernel 1 execution
    for (int i = 0; i < niters; i++) {
        kernel<<<block, threads, sizeof(p) * 1024 * 2>>>(xin_dev, xout_dev, npart, dt, val);

        // Exchange pointers
        x_dev = xin_dev;
        xin_dev = xout_dev;
        xout_dev = x_dev;
    }

    checkCudaErrors(cudaMemcpy(xout, xout_dev, sizeof(p) * npart, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(xin_dev));
    checkCudaErrors(cudaFree(xout_dev));

    // STOP measure time
    cudaEventRecord(stop, 0);

    int count = 0;
    while(count < npart){
        cout << "xout[" << count << "].x: " << xout[count].x << "\n";
        count++;
    }

    // Calculate time
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&execution_time, start, stop);
    printf("kernel execution: %f seconds\n", (execution_time / 1000.0f));
}


int exercise04(int npart, int niters) {
    p *xin = (p *)malloc(sizeof(p) * npart);
    p *xout = (p *)malloc(sizeof(p) * npart);
    init(xin, npart);
    execute_kernel(xin, xout, npart, niters);
    free(xin);
    free(xout);

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

