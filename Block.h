// Block.h
// Ahmed M. Hussein (amhussein4@gmail.com)
// 07/26/2020

#ifndef BLOCK_H_
#define BLOCK_H_

namespace Cascient
{
	enum BlockType
	{
		NullBlockType = 0,
		FeedForwardBlock = 1,
		ConvolutionBlock = 2
	};

	class Block
	{
	public:
		Block();
		~Block();
		virtual void Reset();
		virtual BlockType Type() const = 0;
		virtual void Build() = 0;
		virtual void Push(double* input) = 0;
		virtual void Pull() = 0;
		void ID(const unsigned int& value);
		unsigned int ID() const;
		virtual unsigned int OutputSize() const = 0;
		const double* Output() const;

	private:
		void Initialize();
		unsigned int id;

	protected:
		Block(const Block& block);
		Block& operator=(const Block& block);
		double* output;
	};

	class FFBlock : public Block
	{
	public:
		FFBlock();
		~FFBlock();
		void Reset();
		BlockType Type() const;
		void Build();
		void Push(double* input);
		void Pull();
		unsigned int OutputSize() const;
		
	private:
		FFBlock(const FFBlock& block);
		FFBlock& operator=(const FFBlock& block);
		void Initialize();
	};

	class CVBlock : public Block
	{
	public:
		CVBlock();
		~CVBlock();
		void Reset();
		BlockType Type() const;
		void Input(const unsigned int& width,const unsigned int& height,const unsigned int& depth);
		void Kernel(const unsigned int& width,const unsigned int& height,const unsigned int& count);
		void KernelStride(const unsigned int& horizontal,const unsigned int& vertical);
		void Activation(const int& activation_type);
		void Pooling(const int& pooling_type,const unsigned int& width,const unsigned int& height);
		void PoolingStride(const unsigned int& horizontal,const unsigned int& vertical);
		void Build();
		void Push(double* input);
		void Pull();
		unsigned int OutputSize() const;
		
	private:
		CVBlock(const CVBlock& block);
		CVBlock& operator=(const CVBlock& block);
		void Initialize();
		void DeallocateArrays();
		unsigned int input_width;
		unsigned int input_height;
		unsigned int input_depth;
		unsigned int output_depth;
		unsigned int kernel_width;
		unsigned int kernel_height;
		unsigned int kernel_count;
		unsigned int horizontal_kernel_stride;
		unsigned int vertical_kernel_stride;
		int activation;
		int pooling;
		unsigned int pooling_width;
		unsigned int pooling_height;
		unsigned int horizontal_pooling_stride;
		unsigned int vertical_pooling_stride;
		unsigned int activation_output_width;
		unsigned int activation_output_height;
		unsigned int pooling_output_width;
		unsigned int pooling_output_height;
		double* biases;
		double* weights;
		double* activation_output;
	};
}

#endif

