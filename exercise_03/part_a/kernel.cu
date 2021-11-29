#include <stdio.h>
#include <helper_cuda.h>
#include "utils.h"

__global__ void kernel_color_background(uchar3 *pos,int width, int height) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;

    while(i < N) {
        pos[i].x=255;
        pos[i].y=255;
        pos[i].z=255;
        i+=off;
    }
}

__global__ void kernel_draw_particles(uchar3 *pos,int width, int height, p *particles) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;

    while (i < MAX_PARTICLES) {
        // This block of variables is only to give
        // more readability and avoid to long lines
        int radius = particles[i].radius;
        int Xc = particles[i].x;
        int Yc = particles[i].y;
        int red = particles[i].red;
        int green = particles[i].green;
        int blue = particles[i].blue;

        // We do not need to iterate the complete area of our window
        // witdh x hieght to find the pixels that belong to a particle
        // with radius (r). We can only check in the "square" surrounding
        // the circular shape of (i) particle to do this process less
        // expensive. The next variales compute the square limits:
        // right, left, bottom, top.
        int limit_r = Xc + radius;
        int limit_l = Xc - radius;
        int limit_t = Yc + radius;
        int limit_b = Yc - radius;

        // This block is to validate if our particle is inside
        // of our witdh x hieght window. In opposite case, we assing
        // a valid limit.
        if (limit_r > width)
            limit_r = width;

        if (limit_l < 0)
            limit_l = 0;

        if (limit_t > height)
            limit_t = height;

        if (limit_b < 0)
            limit_b = 0;

        // Iterate our "square" to find the pixel that belong
        // to our circular particle.
        for (int x = limit_l; x <= limit_r; x++) {
            for (int y = limit_b; y <= limit_t; y++) {
                int X = (x - Xc);
                int Y = (y - Yc);
                if (X * X + Y * Y <= radius * radius) {
                    // Conver x,y postion to a index for pos* and
                    // color that pixel.
                    int window_pos = (y * height) - (width - x);
                    if (window_pos < N) {
                        pos[window_pos].x=red;
                        pos[window_pos].y=green;
                        pos[window_pos].z=blue;
                    }
                }
            }
        }

        i+=off;
    }
}

__global__ void kernel_update_particles_pos(uchar3 *pos,int width, int height, p *particles) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    float delta = 0.05;

    while (i < MAX_PARTICLES) {
        // Update particle position (NOTE: Negative gravity aceleration because
        // we want particles go downward).
        particles[i].v0 = particles[i].v0 + (-GRAVITY * delta);
        particles[i].y = particles[i].y + particles[i].v0 * delta;

        // If particle is out of the window size, then set default values
        if (particles[i].y < 0) {
            particles[i].y = particles[i].default_y;
            particles[i].v0 = particles[i].default_v0;
        }

        i+=off;
    }
}


void simulate_p1(uchar3 *ptr, int tick, int w, int h, p *particles)
{
	cudaError_t err=cudaSuccess;

	// set number of threads/blocks
	dim3 block(8, 1, 1);
	dim3 threads(1024, 1, 1);

    p *particles_dev;

	// Call kernel to color background
	kernel_color_background<<< block, threads>>> (ptr, w, h);

    checkCudaErrors(cudaMalloc((void **)&particles_dev, sizeof(p) * MAX_PARTICLES));
    checkCudaErrors(cudaMemcpy(particles_dev, particles, sizeof(p) * MAX_PARTICLES, cudaMemcpyHostToDevice));

    // Call kernel to draw particles
	kernel_draw_particles<<<block, threads>>> (ptr, w, h, particles_dev);

    // Call kernel to update particle values
    kernel_update_particles_pos<<<block, threads>>> (ptr, w, h, particles_dev);
    checkCudaErrors(cudaMemcpy(particles, particles_dev, sizeof(p) * MAX_PARTICLES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(particles_dev));

    // End kernel calls
	err=cudaGetLastError();
	if(err!=cudaSuccess) {
		fprintf(stderr,"Error executing the kernel - %s\n",
				 cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	//  stop in the GL-Loop to look at picture
	if(tick>=1) {
		//getchar();
	}
}
