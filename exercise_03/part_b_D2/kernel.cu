#include <stdio.h>
#include <helper_cuda.h>
#include "utils.h"

__global__ void kernel_draw_background(uchar3 *pos,int width, int height, int tick, p *earth) {
	int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;

    while(i < N) {
        int currentX = i % width;
        int currentY = i / width;

        int Xs = (currentX - SUN_POS_X);
        int Ys = (currentY - SUN_POS_Y);
        //int Xe = (currentX - EARTH_POS_X);
        int Xe = (currentX - earth->x);
        //int Ye = (currentY - EARTH_POS_Y);
        int Ye = (currentY - earth->y);

        // Draw sun
        if (Xs * Xs + Ys * Ys <= SUN_RADIUS * SUN_RADIUS) {
            pos[i].x=255;
            pos[i].y=140;
            pos[i].z=0;

        // Draw earth
        } else if (Xe * Xe + Ye * Ye <= earth->radius * earth->radius) {
            pos[i].x=70;
            pos[i].y=130;
            pos[i].z=180;

        // Color background
        } else {
            pos[i].x=0;
            pos[i].y=0;
            pos[i].z=0;
        }

        i+=off;
    }
}


__global__ void kernel_draw_sun_particles(uchar3 *pos,int width, int height, p *particles) {
	int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;

    while (i < MAX_PARTICLES) {
        int window_pos = (particles[i].y * height) - (width - particles[i].x);
        if (window_pos < N) {
            pos[window_pos].x=255;
            pos[window_pos].y=255;
            pos[window_pos].z=0;
        }
        i+=off;
    }

}


__global__ void kernel_update_particles_pos(uchar3 *pos,int width, int height, p *particles, p *earth) {
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    float time_step = 0.01;
    float ax = 0.0;
    float ay = 0.0;
    //double CONST = -0.015;
    double CONST = 0.015;

    earth->degree = earth->degree + 0.001;
    earth->x = ORBIT_POS_X + ORBIT_RADIUS * cos(earth->degree);
    earth->y = ORBIT_POS_Y + ORBIT_RADIUS * sin(earth->degree);

    if (earth->degree > 360.0)
        earth->degree = 0.0;

    while (i < MAX_PARTICLES) {
        double dx = abs(particles[i].x - earth->x);
        double dy = abs(particles[i].y - earth->y);
        double r = sqrt((dx * dx) + (dy * dy));

        // Aceleration
        ax = ((CONST * earth->mass) / (particles[i].mass * r)) * (particles[i].x / r);
        ay = ((CONST * earth->mass) / (particles[i].mass * r)) * (particles[i].y / r);

        particles[i].vx0 = particles[i].vx0 + (ax * time_step);
        particles[i].vy0 = particles[i].vy0 + (ay * time_step);
        particles[i].x = particles[i].x + particles[i].vx0;
        particles[i].y = particles[i].y + particles[i].vy0;

        // Validate if particle is out of our window width x height
       if (particles[i].y < 0 || particles[i].y > height || particles[i].x < 0 || particles[i].x > width) {
            particles[i].x = particles[i].default_x;
            particles[i].y = particles[i].default_y;
            particles[i].vx0 = particles[i].default_vx0;
            particles[i].vy0 = particles[i].default_vy0;
        }

        i+=off;
    }
}


void simulate(uchar3 *ptr, int tick, int w, int h, p *particles, p *earth)
{
	cudaError_t err=cudaSuccess;

	// set number of threads/blocks
	dim3 block(8,1,1);
	dim3 threads(1024,1,1);

    p *particles_dev;
    p *earth_dev;

	// Call kernel to draw sun, earth and color background
	//
    checkCudaErrors(cudaMalloc((void **)&earth_dev, sizeof(p)));
    checkCudaErrors(cudaMemcpy(earth_dev, earth, sizeof(p), cudaMemcpyHostToDevice));



	kernel_draw_background<<<block, threads>>> (ptr, w, h, tick, earth_dev);
    checkCudaErrors(cudaMalloc((void **)&particles_dev, sizeof(p) * MAX_PARTICLES));
    checkCudaErrors(cudaMemcpy(particles_dev, particles, sizeof(p) * MAX_PARTICLES, cudaMemcpyHostToDevice));

    // Call kernel to draw particles launched by the sun
    kernel_draw_sun_particles<<<block, threads>>> (ptr, w, h, particles_dev);

    // Call kernel to update values for the particles
    kernel_update_particles_pos<<<block, threads>>> (ptr, w, h, particles_dev, earth_dev);
    checkCudaErrors(cudaMemcpy(earth, earth_dev, sizeof(p), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles, particles_dev, sizeof(p) * MAX_PARTICLES, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(particles_dev));
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
