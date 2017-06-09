#ifndef CUDA_HELPERS_H

#pragma once

#pragma warning(push,0)
#include <vector>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <functional>
#pragma warning(pop)

namespace cudaHelpers
{
	template <typename T>
	T* initOnDevice(size_t count)
	{
		T* d_buffer;
		auto size = count * sizeof(T);
		checkCudaErrors(cudaMalloc(&d_buffer, size));
		checkCudaErrors(cudaMemset(d_buffer, 0, size));
		return d_buffer;
	}

	template <typename T>
	T* copyToDevice(std::vector<T> const& vec)
	{
		T* d_buffer = initOnDevice<T>(vec.size());
		checkCudaErrors(cudaMemcpy(d_buffer, &vec[0], vec.size() * sizeof(T), cudaMemcpyHostToDevice));
		return d_buffer;
	}

	template <typename T>
	std::vector<T> copyFromDevice(T const* data, size_t const length)
	{
		std::vector<T> result(length, 42.f);
		checkCudaErrors(cudaMemcpy(&result[0], data, length * sizeof(T), cudaMemcpyDeviceToHost));
		return result;
	}

	template <typename T>
	void freeDeviceMemory(T* data)
	{
		checkCudaErrors(cudaFree(data));
	}
}

#endif