#include <stdio.h>
#include <helper_cuda.h>
 
__global__ void kernel(uchar3 *pos,int width, int height, int tick) {
    // Implement the kernel here
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;

    while(i < N) {
		int currentWid = i % width;
		int currentHei = i / width;
		if(currentWid * currentWid + currentHei * currentHei <= width * height){
			pos[i].x=0;
        	pos[i].y=0;
        	pos[i].z=255;
		}
		else{
			pos[i].x=255;
        	pos[i].y=0;
        	pos[i].z=0;
		}

		int checker = (currentWid / 32 + currentHei / 32) % 2;
		if(tick % 2 == 0){
			checker = 1 - checker;
		}

		if(checker == 0){
			pos[i].x=0;
        	pos[i].y=255;
        	pos[i].z=0;
		}
        i+=off;
    }
	
}
void simulate(uchar3 *ptr, int tick, int w, int h)
{
	cudaError_t err=cudaSuccess;

	// set number of threads/blocks
	dim3 block(1,1,1);
	dim3 threads(1,1,1);

	// call your kernel
	kernel<<< block,threads>>> (ptr,w,h, tick);
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
