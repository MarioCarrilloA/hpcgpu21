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

#endif

