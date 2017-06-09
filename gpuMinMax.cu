#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudaHelpers.hpp"

__global__ void gpuMinMaxOfEachBlock(float const* const input, float* const output, const int cols)
{
	const auto globalThreadPos2d = make_int2(blockIdx.x * blockDim.x + threadIdx.x,	blockIdx.y * blockDim.y + threadIdx.y);
	const auto globalThreadPos1d = globalThreadPos2d.y * cols + globalThreadPos2d.x;

	extern __shared__ float temp[];
	auto minTemp = temp;
	auto maxTemp = temp + blockDim.x;
	minTemp[threadIdx.x] = input[globalThreadPos1d];
	maxTemp[threadIdx.x] = input[globalThreadPos1d];

	// --- Before going further, we have to make sure that all the shared memory loads have been completed
	__syncthreads();

	// --- Reduction in shared memory. Only half of the threads contribute to reduction.
	for (auto s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			minTemp[threadIdx.x] = fminf(minTemp[threadIdx.x], minTemp[threadIdx.x + s]);
			maxTemp[threadIdx.x] = fmaxf(maxTemp[threadIdx.x], maxTemp[threadIdx.x + s]);
		}
		// --- At the end of each iteration loop, we have to make sure that all memory operations have been completed
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		output[blockIdx.y] = minTemp[threadIdx.x];
		output[blockIdx.y + gridDim.y] = maxTemp[threadIdx.x];
	}
}

std::vector<float> gpuMinMaxImpl(std::vector<float> input, int rows, int cols)
{
	auto deviceInput = cudaHelpers::copyToDevice<float>(input);

	auto outputSize = rows * 2;
	auto deviceBlockOutputs = cudaHelpers::initOnDevice<float>(outputSize);

	dim3 blocksInGrid(1, rows);
	dim3 threadsInBlock(cols);
	int sMemSize = threadsInBlock.x * sizeof(float) * 2;

	gpuMinMaxOfEachBlock<<<blocksInGrid, threadsInBlock, sMemSize>>>(deviceInput, deviceBlockOutputs, threadsInBlock.x);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	return cudaHelpers::copyFromDevice(deviceBlockOutputs, outputSize);
}

std::vector<float> gpuMinMax(std::vector<float> input, int rows, int cols)
{
	auto intermediate = gpuMinMaxImpl(input, rows, cols);
	return gpuMinMaxImpl(intermediate, 1, rows*2);
}

