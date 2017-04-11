#pragma warning(push,0)
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#pragma warning(pop)

TEST(ItFails, ExampleTests)
{
	FAIL();
}

int main(int ac, char* av[])
{
	checkCudaErrors(cudaSetDevice(0));
	
	testing::InitGoogleTest(&ac, av);

	return RUN_ALL_TESTS();
}
