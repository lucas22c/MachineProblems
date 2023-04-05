#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <math.h>
#include <ctime>

#define S 250

void HMMulti(int *matrixA, int*matrixB, int *matrixC, int size)
{
  int shift1, shift2;
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
        float hold = 0;
    for (int l = 0; l < size; l++)
    {
        shift1 = i*size + l;
        shift2 = l*size + j;
        hold = hold + matrixA[shift1] + matrixB[shift2];
    }
    matrixC[i*size + j] = hold;
   }
  }
}

__global__ void DMMulti(int *matrixA, int *matrixB, int *matrixC, int size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < size && col < size)
	{
		float hold = 0;
		for(int i = 0; i < size; i++)
    {
        hold = hold + matrix[row*size + i] * matrixB[i * size + col];
      matrixC[row*size + col] = hold;
    }
	}
}

int main()
{
  int main()
{

	cudaEvent_t start_4, stop_4, start_1, start_2, start_3, stop_1, stop_2, stop_3;

	cudaEventCreate(&start_4);
	cudaEventCreate(&start_1);
	cudaEventCreate(&start_2);
	cudaEventCreate(&start_3);

	cudaEventCreate(&stop_4);
	cudaEventCreate(&stop_1);
	cudaEventCreate(&stop_2);
	cudaEventCreate(&stop_3);

	cudaDeviceSynchronize(); //events for start and stop times

	float timer4_gpu = 0.0f, timer1_gpu = 0.0, timer2_gpu = 0.0f, timer3_gpu = 0.0f;
    
  srand ((unsigned)time(&timer));

	size_t hostSize = D*D*sizeof(int); //matrix size calc
    
  int *hosta = (int*)malloc(size); 
	int *hostb = (int*)malloc(size);
	int *hostc = (int*)malloc(size);
  int *hostp = (int*)malloc(size);
    
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
 
int *dMatrixA, *dMatrixB, *dMatrixC;
cudaMalloc((void**)&dMatrixA, hostSize);
cudaMalloc((void**)&dMatrixB, hostSize);
cudaMalloc((void**)&dMatrixC, hostSize);
    
cudaEventRecord(start_1, 0);
cudaMemcpy(dMatrixA, hosta, hostSize, cudaMemcpyHostToDevice);
cudaMemcpy(dMatrixB, hostb, hostSize, cudaMemcpyHostToDevice);
cudaEventRecord(stop_1, 0);
cudaEventSynchronize(stop_1);
cudaEventElapsedTime(&timer1_gpu, start_1, stop_1);
printf("Time to transfer from host to device:%0.2f\n", timer1_gpu);

cudaEventRecord(start_2, 0);
cudaMemcpy(hosta, dMatrixA, hostSize, cudaMemcpyHostToDevice);
cudaMemcpy(hostb, dMatrixB, hostSize, cudaMemcpyHostToDevice);
cudaEventRecord(stop_2, 0);
cudaEventSynchronize(stop_2);
cudaEventElapsedTime(&timer2_gpu, start_2, stop_2);
printf("Time to transfer from device to host:%0.2f\n", timer2_gpu);
    
dim3 threads(S, S, 1);
dim3 blocks(ceil(S / threads.x), ceil(S / threads.y), 1);
    
cudaEventRecord(start_3, 0);
DMMulti <<< blocks, threads >>> (dMatrixA, dMatrixB, dMatrixC, S);
cudaEventRecord(stop_3, 0);
cudaEventSynchronize(stop_3);
cudaEventElapsedTime(&timer3_gpu, start_3, stop_3);
printf("Time taken for GPU matrix multiplication: %0.2f\n", timer3_gpu);
    
cudaEventRecord(start_4, 0);
HMMulti(hosta, hostb, hostp, S);
cudaEventRecord(stop_4, 0);
cudaEventSynchronize(stop_4);
cudaEventElapsedTime(&timer4_gpu, start_4, stop_4);
printf("Time taken for CPU matrix multiplication: %0.2f\n", timer4_gpu);
}
