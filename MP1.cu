//Lucas Coster
//20223016
#include "cuda_runtime.h"
#include <string.h>
#include <stdio.h>


int main()
{
	int count;
	cudaGetDeviceCount(&count);

	printf("Number of devices:%d\n", count);

	for (int i = 0; i < count; i++)
	{
		cudaDeviceProp dp;
		cudaGetDeviceProperties(&dp, i);
		int cores = 0;
		int major = dp.major;
		int mpc = dp.multiProcessorCount;
		switch (major)
		{
		case 2:
			cores = 32 * mpc;
			break;
		case 3:
			cores = 192 * mpc;
			break;
		case 5:
			cores = 128 * mpc;
			break;
		case 6:
			cores = 64 * mpc;
			break;
		case 7:
			cores = 64 * mpc;
			break;
		case 8:
			cores = 64 * mpc;
			break;
		default:
			cores = -1;
			break;
		}
		printf("Device number: %d\n", i);
		printf("Device name and type: %s\n", dp.name);
		printf("Clock Rate: %d\n", dp.clockRate);
		printf("Number of multi-processors: %d\n", dp.multiProcessorCount);
		if (cores == -1) printf("Error getting number of cores");
		else printf("Number of cores: %d\n", cores);
		printf("Warp size: %d\n", dp.warpSize);
		printf("Global memory: %d\n", dp.totalGlobalMem);
		printf("Constant memory: %d\n", dp.totalConstMem);
		printf("Shared memory per block: %d\n", dp.sharedMemPerBlock);
		printf("Registers avaliable per block: %d\n", dp.regsPerBlock);
		printf("Max number of threads per block: %d\n", dp.maxThreadsPerBlock);
		for (int i = 0; i < 3; i++)
		{
			printf("Maximum dimension %d of block: %d\n", i, dp.maxThreadsDim[i]);
		}

		for (int i = 0; i < 3; i++)
		{
			printf("Maximum dimension %d of grid: %d\n", i, dp.maxGridSize[i]);
		}

	}
}