//Lucas Coster
//20223016
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <ctime>

#define D 125	//dimensions
#define T 16	//threads per block

void addition_of_host(int *matrixA, int *matrixB, int*matrixC, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int c = 0; c < size; c++)
		{
			matrixC[i * size + c] = matrixA[i * size + c] + matrixB[i * size + c];
		}
	}
}

__global__ void matrixAdd(int *matrixA, int *matrixB, int *matrixC, int size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < size && col < size)
	{
		int hold = row * size + col;
		matrixC[hold] = matrixA[hold] + matrixB[hold];
	}
}

__global__ void rowAdd(int *matrixA, int *matrixB, int *matrixC, int size)
{
	int l = blockIdx.y * blockDim.y + threadIdx.y;
	int r;
	if (l < size)
	{
		for (int i = 0; i < size; i++)
		{
			r = l * size + i;
			matrixC[r] = matrixB[r] + matrixA[r];
		}
	}
}

__global__ void colAdd(int *dMatrixA, int *dMatrixB, int *dMatrixC, int size)
{
	int l = blockIdx.y * blockDim.y + threadIdx.y;
	int r;
	if (l < size)
	{
		for (int i = 0; i < size; i++)
		{
			r = l * size + i;
			dMatrixC[r] = dMatrixB[r] + dMatrixA[r];
		}
	}
}

int main()
{
	time_t timer; //timer

	int flag1, flag2, flag3; // correct flags

	cudaEvent_t start, stop, start_1, start_2, start_3, stop_1, stop_2, stop_3;

	cudaEventCreate(&start);
	cudaEventCreate(&start_1);
	cudaEventCreate(&start_2);
	cudaEventCreate(&start_3);

	cudaEventCreate(&stop);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&stop_2);
	cudaEventCreate(&stop_3);

	cudaDeviceSynchronize(); //events for start and stop times

	float timer_gpu = 0.0f, timer1_gpu = 0.0, timer2_gpu = 0.0f, timer3_gpu = 0.0f;

	size_t size = D*D*sizeof(int); //matrix size calc

	int *hosta = (int*)malloc(size); //pointers
	int *hostb = (int*)malloc(size);
	int *hostc = (int*)malloc(size);
	int *hostc1 = (int*)malloc(size);
	int *hostc2 = (int*)malloc(size);
	int *hostc3 = (int*)malloc(size);

	int *dMatrixA, *dMatrixB, *dMatrixC; // more pointers
	cudaMalloc((void**)&dMatrixA, size);
	cudaMalloc((void**)&dMatrixB, size);
	cudaMalloc((void**)&dMatrixC, size);

	srand((unsigned)time(&timer)); //seed

	for (int i = 0; i < D; i++)
	{
		for (int k = 0; k < D; k++)
		{
			int randomFirst = rand() % 10;
			int randomSecond = rand() % 10;
			*(hosta + i * D + k) = randomFirst;
			*(hostb + i * D + k) = randomSecond;
		}
	}

	cudaEventRecord(start, 0); // host addition
	addition_of_host(hosta, hostb, hostc, D);
	cudaEventRecord(stop, 0);

	cudaEventElapsedTime(&timer_gpu, start, stop); //results
	printf("Addition of host time: %0.2f\n", timer_gpu);

	cudaMemcpy(dMatrixA, hosta, size, cudaMemcpyHostToDevice); // host matrices to device
	cudaMemcpy(dMatrixB, hostb, size, cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(T, T); //thread setup
	dim3 numberOfBlocks(ceil(D / threadsPerBlock.x), ceil(D / threadsPerBlock.y));

	cudaEventRecord(start_1, 0); // individual threads
	matrixAdd <<< numberOfBlocks, threadsPerBlock >>> (dMatrixA, dMatrixB, dMatrixC, D);
	cudaEventRecord(stop_1, 0);
	cudaEventSynchronize(stop_1);
	cudaMemcpy(hostc1, dMatrixC, size, cudaMemcpyDeviceToHost);
	cudaEventElapsedTime(&timer1_gpu, start_1, stop_1);
	printf("\n matrix addition: %0.2f\n", timer1_gpu);

	cudaEventRecord(start_2, 0); // addition by rows
	rowAdd <<<ceil(D / T), T >>>(dMatrixA, dMatrixB, dMatrixC, D);
	cudaEventRecord(stop_2, 0);
	cudaEventSynchronize(stop_2);
	cudaMemcpy(hostc2, dMatrixC, size, cudaMemcpyDeviceToHost);
	cudaEventElapsedTime(&timer2_gpu, start_2, stop_2);
	printf("\n Time for Row addition: %0.2f\n", timer2_gpu);

	cudaEventRecord(start_3, 0); // addition by columns
	colAdd<<<ceil(D / T), T>>>(dMatrixA, dMatrixB, dMatrixC, D);
	cudaEventRecord(stop_3, 0);
	cudaEventSynchronize(stop_3);
	cudaMemcpy(hostc3, dMatrixC, size, cudaMemcpyDeviceToHost);
	cudaEventElapsedTime(&timer3_gpu, start_3, stop_3);
	printf("\n Timer for column addition: %0.2f\n", timer3_gpu);

	for (int i = 0; i < D; i++) // check correctness
	{
		for (int j = 0; j < D; j++)
		{
			if (*(hostc1 + i * D + j) != *(hostc + i * D + j))
				flag1 = 1;
			if (*(hostc2 + i * D + j) != *(hostc + i * D + j))
				flag2 = 1;
			if (*(hostc3 + i * D + j) != *(hostc + i * D + j))
				flag3 = 1;
		}
	}

	if (flag1 == 1)
		printf("Matrix Addition passed\n");
	else
		printf("Matrix Addition failed\n");
	if (flag2 == 1)
		printf("Row Addition passed\n");
	else
		printf("Row Addition failed\n");
	if (flag3 == 1)
		printf("Column Addition passed\n");
	else
		printf("Column Addition failed\n");

	cudaFreeHost(hosta); // free hosts
	cudaFreeHost(hostb);
	cudaFreeHost(hostc);
	cudaFreeHost(hostc1);
	cudaFreeHost(hostc2);
	cudaFreeHost(hostc3);

	cudaFree(dMatrixA); // free devices
	cudaFree(dMatrixB);
	cudaFree(dMatrixC);

}