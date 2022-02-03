#ifndef KERNEL_EXERCISE_9_H
#define KERNEL_EXERCISE_9_H

struct p {
    float *x;
    float *y;
    float *z;
    float *m;
};

// Kernel 1: Particle simulation exercise 4/7
__global__ void kernel1(p xin, p xout, long int npart, double dt, double val);

// Kernel 2: calculate F
__global__ void kernel2(p xin, float *F, int npart, int M);

// Kernel 2: calculate E
__global__ void kernel3(p xin_now, p xin_old, float *E, int npart);

#endif

