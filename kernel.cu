#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>

__global__ void kernel(float const* const input, float* const output, const int cols)
{
	const auto thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,	blockIdx.y * blockDim.y + threadIdx.y);
	const auto thread_1D_pos = thread_2D_pos.y * cols + thread_2D_pos.x;

	output[thread_1D_pos] = input[thread_1D_pos];
}

void kernelWrapper(float const* const input, float* const output, const int rows, const int cols)
{
	dim3 dimBlock(cols);
	dim3 dimGrid(1, rows);
	kernel<<<dimGrid, dimBlock>>>(input, output, cols);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
