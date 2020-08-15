// Cascient.cpp
// Ahmed M. Hussein (amhussein4@gmail.com)
// 07/26/2020

#include "Cascient.h"
#include "Block.h"
#include "stdio.h"
#include "Random.h"

int main(int argc,char** argv)
{
	Cascient::CVBlock CL1;
	CL1.Input(256,380,3);
	CL1.Kernel(5,5,8);
	CL1.KernelStride(4,4);
	CL1.Activation(1);
	CL1.Pooling(1,3,3);
	CL1.PoolingStride(2,2);
	CL1.Build();

	unsigned int input_size = 256*380*3;
	double* input = new double[input_size];
	for(unsigned int i = 0 ; i < input_size ; i++)
	{
		input[i] = EZ::Random::Uniform();
	}
	CL1.Push(input);
	unsigned int output_size = CL1.OutputSize();
	const double* output = CL1.Output();
	for(unsigned int i = 0 ; i < output_size ; i++)
	{
		if(i%1000 == 0)		printf("%e\n",output[i]);
	}
	delete [] input;
	return 0;
}

