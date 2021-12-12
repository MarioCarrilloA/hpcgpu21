#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <iomanip>
#include <iostream>

#define GB_CONST 1073741824.0

using namespace std;


int main()
{
    cudaError_t err=cudaSuccess;
    cudaEvent_t start,stop;
    float elapsedtime;
    int major = 0, minor = 0;
    int deviceCount = 0;
    //char deviceName[256];

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    /*implement seelction of device and print some information */
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    printf("Selected card %s with capability %d.%d\n", prop.name, major, minor);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate and initialize all needed variables and default values
    long long int iter = 23;
    long long int maximum = powf(2.0, (float)(iter + 1)) / sizeof(float);
    printf("-----------------------------------------------------------------------------\n");
    printf("CUDA Memcpy Power 2 \n");
    printf("Maximal elements (32bit float): %lu\n", maximum);
    printf("Iterations: %lu\n", iter);
    printf("Increment: Pow 2\n");
    printf("-----------------------------------------------------------------------------\n");
    printf("  No. |  repeat k   |  Size bytes |  AVG (ms)   |   Best (ms) |BW (Best GB/s)|\n");
    printf("-----------------------------------------------------------------------------\n");

    float *data_dev;
    float *data_host;
    int k = iter * 100;

    for(int i = 1; i <= iter; i++) {
        double host_dev_best_time = 0.0;
        double dev_host_best_time = 0.0;
        double host_dev_time_acc = 0.0;
        double dev_host_time_acc = 0.0;
        double host_dev_best_bw = 0.0;
        double dev_host_best_bw = 0.0;

        // Variables for bandwidth
        double secs;
        double gbytes;

        /* Allocate memory and calculate size */
        long long int N = powf(2.0, (float)(i + 1));
        data_host = (float *)malloc(N);
        checkCudaErrors(cudaMalloc((void **)&data_dev, N));

        for (int j = 0; j <= k; j++) {
            cudaEventRecord(start);
            // -------- Make benchmark to device copy --------
            checkCudaErrors(cudaMemcpy(data_dev, data_host, N, cudaMemcpyHostToDevice));
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedtime, start, stop);
            host_dev_time_acc += elapsedtime;

            // Ignore first result
            if (j > 0) {
                // Detect best time and compute best bandwidth Host to Dev
    		    if (host_dev_best_time == 0 || host_dev_best_time > elapsedtime) {
                    host_dev_best_time = elapsedtime;
                    gbytes = N / GB_CONST;
                    secs = (elapsedtime / 1000.0f);
                    host_dev_best_bw = gbytes / secs;
                }
            }

            cudaEventRecord(start);
            // -------- Make benchmark to host copy --------
            checkCudaErrors(cudaMemcpy(data_host, data_dev, N, cudaMemcpyDeviceToHost));
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsedtime, start, stop);
            dev_host_time_acc = elapsedtime;

            // Ignore first result
            if (j > 0) {
                // Detect best time and compute best bandwidth Dev to Host
    	        if (dev_host_best_time == 0 || dev_host_best_time > elapsedtime) {
                    dev_host_best_time = elapsedtime;
                    gbytes = N / GB_CONST;
                    secs = (elapsedtime / 1000.0f);
                    host_dev_best_bw  = gbytes / secs;
                }
            }
        }

        // Compute average HOST to DEV
        double host_dev_avg_time = host_dev_time_acc / k;

        // Compute average DEV to HOST
        double dev_host_avg_time = dev_host_time_acc / k;

        // Print results
        cout << setw(10) << left << i << left;
        cout << setw(15) << k << left;
        cout << setw(12) << N << left;
        cout << setw(14) << host_dev_avg_time << left;
        cout << setw(13) << host_dev_best_time << left;
        cout << setw(13) << host_dev_best_bw << left;
        cout << endl;
        k = k - 100;

        // free memory
        free(data_host);
        checkCudaErrors(cudaFree(data_dev));
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // uncomment using Windows getchar();

    return 0;
}
