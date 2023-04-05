// Lucas Coster
// 20223016

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

#define B 16
#define T 10
#define Dim 125

void HostMatrixMultiplication(float *matrixA, float *matrixB, float *matrixC, int size)
{
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			float hold = 0;
			for (int l = 0; l < size; l++)
			{
				int shift1 = i + size + l;
				int shift2 = l*size + j;

				hold = hold + matrixA[shift1] * matrixB[shift2];
			}
			matrixC[i*size + j] = hold;
		}
	}
}

__global__ void DeviceMatrixMultiplication(float *matrixA, float *matrixB, float *matrixC, int n, int width)
{
	__shared__ float mds[T][T];
	__shared__ float nds[T][T];

	int i = blockIdx.x * T + threadIdx.x;
	int j = blockIdx.y * T + threadIdx.y;

	int idx = i + j * n;
	float P = 0;
	for (int c = 0; c < n / T; c++)
	{
		if (i < n && j < n)
		{
			mds[threadIdx.y][threadIdx.x] = matrixA[j * n + (c * T + threadIdx.x)];
			nds[threadIdx.y][threadIdx.x] = matrixB[i + (c * T + threadIdx.y)];
			__syncthreads();

			for (int r = 0; r < T; r++)
			{
				P += mds[threadIdx.y][r] * nds[r][threadIdx.x];
				__syncthreads();
			}
		}
	}
	matrixC[idx] = P;
}

int main()
{
	cudaEvent_t start_1, stop_1;
	cudaEventCreate(&start_1);
	cudaEventCreate(&stop_1);

	time_t timer;

	float timer1_gpu;

	cudaEventSynchronize;

	srand((unsigned)time(&timer));
	size_t size;
	size = Dim*Dim*sizeof(int);

		float *hosta = (float*)malloc(size);
		float *hostb = (float*)malloc(size);
		float *hostc = (float*)malloc(size);

		for (int i = 0; i < Dim; i++)
		{
			for (int j = 0; j < Dim; j++)
			{
				float randomFirst = rand() % 10;
				float randomSecond = rand() % 10;
				*(hostb + i * Dim + j) = randomFirst;
				*(hostb + i * Dim + j) = randomSecond;
			}
		}

		float *dMatrixA, *dMatrixB, *dMatrixC;

		cudaMalloc((void**)&dMatrixA, Dim*Dim*sizeof(float));
		cudaMalloc((void**)&dMatrixB, Dim*Dim*sizeof(float));
		cudaMalloc((void**)&dMatrixC, Dim*Dim*sizeof(float));

		cudaMemcpy(dMatrixA, hosta, (Dim*Dim)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dMatrixB, hostb, (Dim*Dim)*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dMatrixC, hostc, (Dim*Dim)*sizeof(float), cudaMemcpyHostToDevice);

		cudaEventRecord(start_1);
		dim3 threads(T, T);
		dim3 blocks(ceil(Dim / T), ceil(Dim / T));

		DeviceMatrixMultiplication << <blocks, threads >> > (dMatrixA, dMatrixB, dMatrixC, Dim, T);

		cudaEventRecord(stop_1);
		cudaEventSynchronize(stop_1);
		cudaEventElapsedTime(&timer1_gpu, start_1, stop_1);
		cudaEventDestroy(stop_1);
		cudaEventDestroy(stop_1);
		printf("Kernel matrix multiplication with %d tiles time: %0.2f\n", T, timer1_gpu);

		cudaMemcpy(hostc, dMatrixC, (Dim*Dim)*sizeof(float), cudaMemcpyDeviceToHost);
		float *hostcTemp = (float*)malloc(size);

		HostMatrixMultiplication(hosta, hostb, hostcTemp, Dim);

		int good = 1;
		for (int i = 0; i < Dim; i++)
		{
			for (int j = 0; j < Dim; j++)
			{
				float diff = *(hostc + i * Dim + j) - *(hostcTemp + i * Dim + j);
				if (diff < 1)
					good = 0;
			}
		}

		if (good == 1) {
			printf("Test Failed\n");
		}
		else {
			printf("Test Passed\n");
		}

		cudaFree(dMatrixA);
		cudaFree(dMatrixB);
		cudaFree(dMatrixC);
}