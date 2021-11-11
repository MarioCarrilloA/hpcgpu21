#include <stdio.h>
#include <helper_cuda.h>
 
__global__ void kernel(uchar3 *pos,int width, int height) {
	// Implement the kernel here
}
void simulate(uchar3 *ptr, int tick, int w, int h)
{
	cudaError_t err=cudaSuccess;

	// set number of threads/blocks
	dim3 block(1,1,1);
	dim3 threads(1,1,1);

	// call your kernel
	kernel<<< block,threads>>> (ptr,w,h);
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
