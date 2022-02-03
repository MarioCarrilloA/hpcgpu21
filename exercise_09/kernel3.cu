#include <stdio.h>
#include <kernels.h>

// E computation. Vector/Matrix multiplication  kernel
__global__ void kernel3(p xin_now, p xin_old, float *E, int npart) {
    extern __shared__ float psum[];
    float dstx;
    float dsty;
    float dstz;
    float m;

    // partial sum per thread
    for (int idx = threadIdx.x; idx < npart; idx+=blockDim.x) {
        dstx = (xin_now.x[idx] - xin_old.x[idx]) * (xin_now.x[idx] - xin_old.x[idx]);
        dsty = (xin_now.y[idx] - xin_old.y[idx]) * (xin_now.y[idx] - xin_old.y[idx]);
        dstz = (xin_now.z[idx] - xin_old.z[idx]) * (xin_now.z[idx] - xin_old.z[idx]);
        m = xin_now.m[idx];

        // Energy calculation
        psum[threadIdx.x] = (m * (dstx + dsty + dstz)) / 2;
    }

    // bringing thread groups together
    __syncthreads();
    int off = blockDim.x/2;

    // start with 1/2 #threads
    while (off > 31) {
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

    *E = psum[threadIdx.x];
}
