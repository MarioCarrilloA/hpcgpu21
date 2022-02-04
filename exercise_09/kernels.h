#ifndef KERNEL_EXERCISE_9_H
#define KERNEL_EXERCISE_9_H

struct p {
    float *x;
    float *y;
    float *z;
    float *m;
};

// Kernel 1: Particle simulation exercise 4/7
__global__ void kernel1(p xin, p xout, long int npart, float *dt, float *val);

// Kernel 2: calculate F
__global__ void kernel2(p xin, float *F, int npart, float M, float *dt, float *val);

// Kernel 2: calculate E
__global__ void kernel3(p xin_now, p xin_old, float *E, int npart);

#endif

