#include <stdio.h>
#include <helper_cuda.h>
#include "utils.h"

__global__ void kernel_draw_earth(uchar3 *pos,int width, int height, int tick, Earth *earth) {
	int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;

	float radians = ((tick / 2) % 360) * (PI / 180);
	earth->xc = width / 2 + 200 * std::cos(radians);
	earth->yc  = height / 2 + 200 * std::sin(radians);

	while(i < N){
		int currentX = i % width;
		int currentY = i / width;
		int distance = (currentX - earth->xc) * (currentX - earth->xc) + (currentY - earth->yc) * (currentY - earth->yc);
		if(distance < earth->radius * earth->radius){
			pos[i].x = 14;
			pos[i].y = 175;
			pos[i].z = 105;
		}
		else{
			pos[i].x = 0;
			pos[i].y = 0;
			pos[i].z = 0;
		}
		i+=off;
	}

}

__global__ void kernel_draw_sun(uchar3 *pos,int width, int height) {
	int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;

	while(i < N){
		int currentX = i % width;
		int currentY = i / width;
		int distance = (currentX - (width / 2)) * (currentX - (width / 2))  
						+ (currentY - (height / 2)) * (currentY - (height / 2));
		if(distance < 900){
			pos[i].x = 239;
			pos[i].y = 142;
			pos[i].z = 56;
		}
		i+=off;
	}

}

__global__ void kernel_draw_particles(uchar3 *pos, int width, int height, p *particles){
	int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;
	
	while(i < MAX_PARTICLES){
		// This block of variables is only to give
        // more readability and avoid to long lines
        int Xc = particles[i].x;
        int Yc = particles[i].y;
        int red = particles[i].red;
        int green = particles[i].green;
        int blue = particles[i].blue;

		if(0 <= Xc && Xc < width && 0 <= Yc && Yc <= height){
			int window_pos = (Yc * height) - (width - Xc);
			pos[window_pos].x = red;
			pos[window_pos].y = green;
			pos[window_pos].z = blue;
		}

		i+=off;
	}
}

__global__ void kernel_update_particles_pos(uchar3 *pos, int width, int height, p *particles, Earth *earth){
	int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    float delta = 0.2;

	while (i < MAX_PARTICLES) {
        // Update particle position 
		float theta = (2 * PI) / MAX_PARTICLES;
		float degree = i * theta;

        particles[i].v0 = particles[i].v0 + (particles[i].mass * delta);
		particles[i].x = particles[i].x + particles[i].v0 * std::cos(degree);
        particles[i].y = particles[i].y + particles[i].v0 * std::sin(degree);

		// Apply a repulsive force of earth
		float force = 50 * (GRAVITY / earth->radius);
		if((earth->xc - earth->radius - force) < particles[i].x && particles[i].x < (earth->xc + earth->radius + force)
			&& (earth->yc - earth->radius - force) < particles[i].y && particles[i].y < (earth->yc + earth->radius + force)){
						
			particles[i].x = earth->xc + (earth->radius + force) * std::cos(degree);
			particles[i].y = earth->yc + (earth->radius + force) * std::sin(degree);
		}

		// Update particle color
		particles[i].red = (particles[i].red + 1) % 256;
		particles[i].green = (particles[i].green + 1) % 256;
		particles[i].blue = (particles[i].blue + 1) % 256;

        // If particle is out of the window size, then set default values
        if (particles[i].y < 0 || particles[i].y >= height || particles[i].x < 0 || particles[i].x >= width) {
			particles[i].x = particles[i].default_x;
            particles[i].y = particles[i].default_y;
            particles[i].v0 = particles[i].default_v0;

			particles[i].red = 100;
			particles[i].green = 70;
			particles[i].blue = 50;
        }

        i+=off;
    }
}

void simulate_p1(uchar3 *ptr, int tick, int w, int h, p *particles, Earth *earth)
{
	cudaError_t err=cudaSuccess;

	// set number of threads/blocks
	dim3 block(8,1,1);
	dim3 threads(1024,1,1);

	p *particles_dev;
	Earth *earth_dev;

	checkCudaErrors(cudaMalloc((void **)&earth_dev, sizeof(Earth)));
	checkCudaErrors(cudaMemcpy(earth_dev, earth, sizeof(Earth), cudaMemcpyHostToDevice));

	// Call kernel to draw earth
	kernel_draw_earth<<<block, threads>>> (ptr,w,h,tick,earth_dev);

	checkCudaErrors(cudaMalloc((void **)&particles_dev, sizeof(p) * MAX_PARTICLES));
	checkCudaErrors(cudaMemcpy(particles_dev, particles, sizeof(p) * MAX_PARTICLES, cudaMemcpyHostToDevice));
	
	// Call kernel to update particle values
    kernel_update_particles_pos<<<block, threads>>> (ptr, w, h, particles_dev, earth_dev);

	// Call kernel to draw particles
	kernel_draw_particles<<<block, threads>>> (ptr, w, h, particles_dev);

	// Call kernel to draw sun
	kernel_draw_sun<<< block,threads>>> (ptr,w,h);

	checkCudaErrors(cudaMemcpy(earth, earth_dev, sizeof(Earth), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(earth_dev));
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
