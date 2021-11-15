#include <stdio.h>
#include <helper_cuda.h>
#define PI 3.1416
 
__global__ void kernel(uchar3 *pos,int width, int height, int tick) {
    // Implement the kernel here
    int t = threadIdx.x;
    int g = blockIdx.x;
    int i = t + g * blockDim.x;
    int off = gridDim.x * blockDim.x;
    int N = width * height;

    // Values for animated circle
    int xc = (tick % width) + 50;
    int yc = 100;
    int radius = 60;

    while (i < N) {
		int currentWid = i % width;
		int currentHei = i / width;
		if(currentWid * currentWid + currentHei * currentHei <= width * height){
			pos[i].x=0;
        	pos[i].y=0;
        	pos[i].z=255;
		}
		else {
			pos[i].x=255;
        	pos[i].y=0;
        	pos[i].z=0;
		}

		int checker = (currentWid / 32 + currentHei / 32) % 2;
		if (tick % 4 < 2){
			checker = 1 - checker;
		}

		if(checker == 0){
			pos[i].x=0;
        	pos[i].y=(tick+150) % 255;
        	pos[i].z=0;
		}

        // 199 and 601 will be the range of sine amplitude
        if (currentHei >= 199 && currentHei <= 601) {
            double degree = (currentWid * 360.0) / 800.0;
            double radians = degree * (PI/180.0);
            double amplitude = sin(radians);
            int pixely = ((amplitude + 1) * 400) / 2;
            // 199, 200 and 201 are bias values to do our line thicker
            if (pixely + 199 == currentHei
                || pixely + 200 == currentHei
                || pixely + 201 == currentHei ) {
                pos[i].x=255;
                pos[i].y=255;
                pos[i].z=0;
            }
        }

        int X = (currentWid - xc);
        int Y = (currentHei - yc);
        if (X * X + Y * Y <= radius * radius) {
            pos[i].x=255;
            pos[i].y=178;
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
