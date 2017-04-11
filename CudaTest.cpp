#include <gtest/gtest.h>
#include <gmock/gmock.h>

TEST(ItFails, ExampleTests)
{
	FAIL();
}

int main(int ac, char* av[])
{
	testing::InitGoogleTest(&ac, av);
	return RUN_ALL_TESTS();
}
