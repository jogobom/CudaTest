#pragma warning(push,0)
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#pragma warning(pop)

template <typename T>
T* AllocOnDevice(size_t count)
{
	T* d_buffer;
	checkCudaErrors(cudaMalloc(&d_buffer, count*sizeof(T)));
	return d_buffer;
}

template <typename T>
T* CopyToDevice(std::vector<T> const& vec)
{
	T* d_buffer = AllocOnDevice<T>(vec.size());
	checkCudaErrors(cudaMemcpy(d_buffer, &vec[0], vec.size() * sizeof(T), cudaMemcpyHostToDevice));
	return d_buffer;
}

template <typename T>
std::vector<T> CopyFromDevice(T const* data, size_t const length)
{
	std::vector<T> result(length, 42.f);
	checkCudaErrors(cudaMemcpy(&result[0], data, length * sizeof(T), cudaMemcpyDeviceToHost));
	return result;
}

std::vector<float> CpuVersion(std::vector<float> const& input)
{
	return input;
}

void kernelWrapper(float const* const input, float* const output, const int rows, const int cols);

TEST(ExampleTests, Identity)
{
	auto rows = 2;
	auto cols = 1;
	auto length = rows * cols;

	std::vector<float> input;
	for (auto i = 0; i < rows; ++i)
		for (auto j = 0; j < cols; ++j)
			input.push_back(static_cast<float>(j + i*cols));

	const auto d_input = CopyToDevice<float>(input);
	const auto d_output = AllocOnDevice<float>(length);

	kernelWrapper(d_input, d_output, rows, cols);

	const auto gpuOutput = CopyFromDevice(d_output, length);

	const auto cpuOutput = CpuVersion(input);

	ASSERT_THAT(gpuOutput, testing::ContainerEq(cpuOutput));
}

int main(int ac, char* av[])
{
	checkCudaErrors(cudaSetDevice(0));
	
	testing::InitGoogleTest(&ac, av);

	return RUN_ALL_TESTS();
}
