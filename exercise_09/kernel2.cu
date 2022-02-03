#include <stdio.h>
#include <kernels.h>

// F computation. Vector/Matrix multiplication
__global__ void kernel2(p xin, float *F, int npart, int M) {
    extern __shared__ float psum[];
    float delta_r = 0.0f;

    // partial sum per thread
    for (int idx = threadIdx.x; idx < npart; idx+=blockDim.x) {
        delta_r = (
                (xin.x[idx] * xin.x[idx]) +
                (xin.y[idx] * xin.y[idx]) +
                (xin.z[idx] * xin.z[idx])
       );

        // Force calculatio
        psum[threadIdx.x] = (M * xin.m[idx]) / delta_r;
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

    *F = psum[threadIdx.x];
}
