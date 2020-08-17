// Block.cpp
// Ahmed M. Hussein (amhussein4@gmail.com)
// 07/26/2020

#include "Block.h"
#include "math.h"
#include "string.h"
#include "Random.h"
#include "BLAS.h"
#include "float.h"

namespace Cascient
{
	Block::Block(){Initialize();}
	Block::Block(const Block& block){*this = block;}
	Block::~Block(){Reset();}
	Block& Block::operator=(const Block& block)
	{
		id = block.id;
		return *this;
	}
	void Block::Reset()
	{
		if(output != 0)
		{
			delete [] output;
			output = 0;
		}
		Initialize();
	}
	void Block::ID(const unsigned int& value){id = value;}
	unsigned int Block::ID() const{return id;}
	const double* Block::Output() const{return output;}
	void Block::Initialize()
	{
		id = 0;
		output = 0;
	}

	FFBlock::FFBlock(){Initialize();}
	FFBlock::FFBlock(const FFBlock& block) : Block(block){*this = block;}
	FFBlock::~FFBlock(){Reset();}
	FFBlock& FFBlock::operator=(const FFBlock& block)
	{
		Block::operator=(block);
		input_size = block.input_size;
		output_size = block.output_size;
		unit_count = block.unit_count;
		activation = block.activation;
		return *this;
	}
	void FFBlock::Reset()
	{
		DeallocateArrays();
		Initialize();
		Block::Reset();
	}
	BlockType FFBlock::Type() const{return FeedForwardBlock;}
	void FFBlock::Build()
	{

	}
	void FFBlock::Push(double* input)
	{

	}
	void FFBlock::Pull()
	{

	}
	unsigned int FFBlock::InputSize() const{return input_size;}
	unsigned int FFBlock::OutputSize() const{return output_size;}
	void FFBlock::Initialize()
	{
		input_size = 0;
		output_size = 0;
		unit_count = 0;
		activation = 0;
		input_gradients = 0;
		output_gradients = 0;
		weights = 0;
	}
	void FFBlock::DeallocateArrays()
	{
		if(input_gradients != 0)		delete [] input_gradients;
		if(output_gradients != 0)		delete [] output_gradients;
		if(weights != 0)				delete [] weights;
		if(output != 0)					delete [] output;
		input_gradients = 0;
		output_gradients = 0;
		weights = 0;
		output = 0;
	}

	CVBlock::CVBlock(){Initialize();}
	CVBlock::CVBlock(const CVBlock& block) : Block(block){*this = block;}
	CVBlock::~CVBlock(){Reset();}
	CVBlock& CVBlock::operator=(const CVBlock& block)
	{
		// equating and copying are prohibited
		Block::operator=(block);
		input_width = block.input_width;
		input_height = block.input_height;
		input_depth = block.input_depth;
		output_depth = block.output_depth;
		kernel_width = block.kernel_width;
		kernel_height = block.kernel_height;
		kernel_count = block.kernel_count;
		horizontal_kernel_stride = block.horizontal_kernel_stride;
		vertical_kernel_stride = block.vertical_kernel_stride;
		activation = block.activation;
		pooling = block.pooling;
		pooling_width = block.pooling_width;
		pooling_height = block.pooling_height;
		horizontal_pooling_stride = block.horizontal_pooling_stride;
		vertical_pooling_stride = block.vertical_pooling_stride;
		activation_output_width = block.activation_output_width;
		activation_output_height = block.activation_output_height;
		pooling_output_width = block.pooling_output_width;
		pooling_output_height = block.pooling_output_height;
		return *this;
	}
	void CVBlock::Reset()
	{
		DeallocateArrays();
		Initialize();
		Block::Reset();
	}
	BlockType CVBlock::Type() const{return ConvolutionBlock;}
	void CVBlock::Input(const unsigned int& width,const unsigned int& height,const unsigned int& depth)
	{
		input_width = width;
		input_height = height;
		input_depth = depth;
	}
	void CVBlock::Kernel(const unsigned int& width,const unsigned int& height,const unsigned int& count)
	{
		kernel_width = width;
		kernel_height = height;
		kernel_count = count;
	}
	void CVBlock::KernelStride(const unsigned int& horizontal,const unsigned int& vertical)
	{
		horizontal_kernel_stride = horizontal;
		vertical_kernel_stride = vertical;
	}
	void CVBlock::Activation(const int& activation_type){activation = activation_type;}
	void CVBlock::Pooling(const int& pooling_type,const unsigned int& width,const unsigned int& height)
	{
		pooling = pooling_type;
		pooling_width = width;
		pooling_height = height;
	}
	void CVBlock::PoolingStride(const unsigned int& horizontal,const unsigned int& vertical)
	{
		horizontal_pooling_stride = horizontal;
		vertical_pooling_stride = vertical;
	}
	void CVBlock::Build()
	{
		DeallocateArrays();
		output_depth = kernel_count;
		// the output size per dimension is 
		// n_out = floor((n_in - kernel_size)/stride) + 1
		activation_output_width = (unsigned int)(floor((input_width - kernel_width)/horizontal_kernel_stride)) + 1;
		activation_output_height = (unsigned int)(floor((input_height - kernel_height)/vertical_kernel_stride)) + 1;
		if(pooling == 0)
		{
			// in the case of no pooling, the pooling and activation 
			// outputs are the same
			pooling_output_width = activation_output_width;
			pooling_output_height = activation_output_height;
		}
		else
		{
			pooling_output_width = (unsigned int)(floor((activation_output_width - pooling_width)/horizontal_pooling_stride)) + 1;
			pooling_output_height = (unsigned int)(floor((activation_output_height - pooling_height)/vertical_pooling_stride)) + 1;
		}
		// allocate and initialize output and weight arrays
		biases = new double[kernel_count];
		weights = new double[kernel_count*kernel_width*kernel_height*input_depth];
		activation_output = new double[activation_output_width*activation_output_height*output_depth];
		output = new double[pooling_output_width*pooling_output_height*output_depth];
		memset(activation_output,0,activation_output_width*activation_output_height*output_depth*sizeof(double));
		memset(output,0,pooling_output_width*pooling_output_height*output_depth*sizeof(double));
		unsigned int kernel_channel_size = kernel_width*kernel_height;
		unsigned int kernel_size = kernel_channel_size*input_depth;
		for(unsigned int k = 0 ; k < kernel_count ; k++)
		{
			biases[k] = EZ::Random::Uniform();
			for(unsigned int j = 0 ; j < kernel_height ; j++)
			{
				for(unsigned int i = 0 ; i < kernel_width ; i++)
				{
					for(unsigned int l = 0 ; l < input_depth ; l++)
					{
						weights[k*kernel_size + j*kernel_channel_size + i*input_depth + l] = EZ::Random::Uniform();
					}
				}
			}
		}
	}
	void CVBlock::Push(double* input)
	{
		// pass the inputs through the convolution process and apply the activations 
		// on the convolution output 
		// all kernel dimensions are assumed to be odd numbers
		unsigned int kernel_channel_size = kernel_width*kernel_height;
		unsigned int kernel_size = kernel_channel_size*input_depth;
		unsigned int input_channel_size = input_width*input_height;
		unsigned int activation_channel_size = activation_output_width*activation_output_height;
		unsigned int input_i = 0;
		unsigned int input_j = 0;
		double convolution = 0.0;
		double activation_input = 0.0;
		for(unsigned int k = 0 ; k < kernel_count ; k++)
		{
			for(unsigned int j = 0 ; j < activation_output_height ; j++)
			{
				// compute the location of the upper left corner of the input 
				// that will be convolved with the kernel
				input_j = j*vertical_kernel_stride;
				for(unsigned int i = 0 ; i < activation_output_width ; i++)
				{
					input_i = i*horizontal_kernel_stride;
					convolution = 0.0;
					for(unsigned int c = 0 ; c < input_depth ; c++)
					{
						for(unsigned int p = 0 ; p < kernel_height ; p++)
						{
							convolution += EZ::Math::BLAS::DotProduct(kernel_width,1,
									&weights[k*kernel_size + c*kernel_channel_size + p*kernel_width],
									1,&input[c*input_channel_size + (input_j + p)*input_width + input_i]);
						}
					}
					// activation is identity by default
					activation_input = convolution + biases[k];
					activation_output[k*activation_channel_size + j*activation_output_width + i] = activation_input;
					if(activation == 1)
					{
						// ReLU: rectified linear unit activation
						if(activation_input < 0.0)		activation_output[k*activation_channel_size + j*activation_output_width + i] = 0.0;
					}
					else if(activation == 2)
					{
						// Sigmoid activation
						activation_output[k*activation_channel_size + j*activation_output_width + i] = 1.0/(1.0 + exp(-activation_input));
					}			
				}
			}
		}
		// apply pooling if required
		if(pooling == 0)
		{
			// in the case of no pooling, copy activation 
			// output to pooling output
			memcpy(output,activation_output,pooling_output_width*pooling_output_height*output_depth*sizeof(double));
			return;
		}
		double pooling_value = 0.0;
		double value = 0.0;
		unsigned int pooling_channel_size = pooling_output_width*pooling_output_height;
		for(unsigned int k = 0 ; k < kernel_count ; k++)
		{
			for(unsigned int j = 0 ; j < pooling_output_height ; j++)
			{
				// compute the location of the upper left corner of the activation  
				// output that will be pooled
				input_j = j*vertical_pooling_stride;
				for(unsigned int i = 0 ; i < pooling_output_width ; i++)
				{
					input_i = i*horizontal_pooling_stride;
					if(pooling == 1)
					{
						// max pooling
						pooling_value = -DBL_MAX;
						for(unsigned int p = 0 ; p < pooling_height ; p++)
						{
							for(unsigned int q = 0 ; q < pooling_width ; q++)
							{
								value = activation_output[k*activation_channel_size + (input_j + p)*activation_output_width + input_i + q];
								if(value > pooling_value)			pooling_value = value;
							}
						}
					}
					else if(pooling == 2)
					{
						// average pooling
						pooling_value = 0.0;
						for(unsigned int p = 0 ; p < pooling_height ; p++)
						{
							for(unsigned int q = 0 ; q < pooling_width ; q++)
							{
								pooling_value += activation_output[k*activation_channel_size + (input_j + p)*activation_output_width + input_i + q];
							}
						}
						pooling_value = pooling_value/(double)(pooling_height)/(double)(pooling_width);
					}
					output[k*pooling_channel_size + j*pooling_output_width + i] = pooling_value;
				}
			}
		}
	}
	void CVBlock::Pull()
	{
		
	}
	unsigned int CVBlock::InputSize() const{return(input_width*input_height*input_depth);}
	unsigned int CVBlock::OutputSize() const{return (pooling_output_width*pooling_output_height*output_depth);}
	void CVBlock::Initialize()
	{
		input_width = 1;
		input_height = 1;
		input_depth = 1;
		output_depth = 0;
		kernel_width = 1;
		kernel_height = 1;
		kernel_count = 1;
		horizontal_kernel_stride = 1;
		vertical_kernel_stride = 1;
		activation = 0;
		pooling = 0;
		pooling_width = 1;
		pooling_height = 1;
		horizontal_pooling_stride = 1;
		vertical_pooling_stride = 1;
		activation_output_width = 0;
		activation_output_height = 0;
		pooling_output_width = 0;
		pooling_output_height = 0;
		biases = 0;
		weights = 0;
		activation_output = 0;
		output = 0;
	}
	void CVBlock::DeallocateArrays()
	{
		if(biases != 0)				delete [] biases;
		if(weights != 0)			delete [] weights;
		if(activation_output != 0)	delete [] activation_output;
		if(output != 0)				delete [] output;
		biases = 0;
		weights = 0;
		activation_output = 0;
		output = 0;
	}
}

