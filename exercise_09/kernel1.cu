#include <stdio.h>
#include <kernels.h>

// Particle simulation exercise 4/7
__global__ void kernel1(p xin, p xout, long int npart, float *dt, float *val) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int m = 2;
    int size = m * blockDim.x;
    float maxrad = 0.9f;
    float f = 0.0;
    float dti = *dt;
    float vali = *val;

    // distance vars
    float dsq;
    float dstx;
    float dsty;
    float dstz;

    extern __shared__ float x_shared[];
    p xj_shared;

    // Split shared memry
    xj_shared.x = &x_shared[0];
    xj_shared.y = &x_shared[blockDim.x * 2];
    xj_shared.z = &x_shared[blockDim.x * 4];
    xj_shared.m = &x_shared[blockDim.x * 8];

    if (i < npart) {
        xout.x[i] = xin.x[i];
        xout.y[i] = xin.y[i];
        xout.z[i] = xin.z[i];

        for(int ja = 0; ja < npart; ja+=size) {

            // compute particles according to block size multiplier
            for(int jl = 0; jl < size/blockDim.x; jl += blockDim.x) {
                int jdx = jl + t;
                int idx = ja + jl + t;

                // Copy to shared memory
                xj_shared.x[jdx] = xin.x[idx];
                xj_shared.y[jdx] = xin.y[idx];
                xj_shared.z[jdx] = xin.z[idx];
                xj_shared.m[jdx] = xin.m[idx];
            }
            __syncthreads();

            for(int j = ja; j < ja + size; j++){
                dstx = xin.x[t] - xj_shared.x[j - ja];
                dsty = xin.y[t] - xj_shared.y[j - ja];
                dstz = xin.z[t] - xj_shared.z[j - ja];

                // Compute distance
                dsq = (dstx * dstx) + (dsty * dsty) + (dstz * dstz);

                if (dsq < maxrad && dsq != 0 && i != j) {
                    f += xin.m[t] * xj_shared.m[j - ja] * (xin.x[t] - xj_shared.x[j - ja]) / dsq;
                }
            }
        }

        float s = f * dti * vali;
        xout.x[i] += s;
        xout.y[i] += s;
        xout.z[i] += s;
    }
}


