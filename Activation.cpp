// Activation.cpp
// Ahmed M. Hussein (amhussein4@gmail.com)
// 08/17/2020

#include "Activation.h"
#include "math.h"
#include "InputOutput.h"
#include "string.h"

ActivationFunction::ActivationFunction(){Initialize();}
ActivationFunction::ActivationFunction(const ActivationFunction& function){*this = function;}
ActivationFunction::~ActivationFunction(){Reset();}
ActivationFunction& ActivationFunction::operator=(const ActivationFunction& function){return *this;}
void ActivationFunction::Reset(){Initialize();}
double ActivationFunction::operator()(const double& argument) const{return Evaluate(argument);}
ActivationFunction* ActivationFunction::ReadAndCreate(FILE* file)
{
	char line[1024];
	if(!EZ::IO::ReadLine(line,1024,file))		return 0;
	ActivationFunction* function = 0;
	if(strncmp(line,"identity",8) == 0)
	{
		fseek(file,-8,SEEK_CUR);
		function = new Identity;
	}
	else if(strncmp(line,"sigmoid",7) == 0)
	{
		fseek(file,-7,SEEK_CUR);
		function = new Sigmoid;
	}
	else if(strncmp(line,"relu",4) == 0)
	{
		fseek(file,-4,SEEK_CUR);
		function = new ReLU;
	}
	else
	{
		printf("unknown word : %s\n",line);
		return 0;
	}
	if(!function->Read(file))
	{
		delete function;
		return 0;
	}
	return function;
}
void ActivationFunction::Initialize(){}

Identity::Identity(){Initialize();}
Identity::Identity(const Identity& function) : ActivationFunction(function){*this = function;}
Identity::~Identity(){Reset();}
Identity& Identity::operator=(const Identity& function)
{
	ActivationFunction::operator=(function);
	return *this;
}
void Identity::Reset(){ActivationFunction::Reset();}
double Identity::Evaluate(const double& argument) const{return argument;}
double Identity::Differentiate(const double& argument) const{return 1.0;}
ActivationFunction* Identity::Clone() const{return new Identity(*this);}
void Identity::Write(FILE* file) const{fprintf(file,"indentity\n");}
bool Identity::Read(FILE* file)
{
	char line[1024];
	if(!EZ::IO::ReadLine(line,1024,file))		return false;
	sscanf(line,"indentity\n");
	return true;
}
void Identity::Initialize(){ActivationFunction::Initialize();}

Sigmoid::Sigmoid(){Initialize();}
Sigmoid::Sigmoid(const Sigmoid& function) : ActivationFunction(function){*this = function;}
Sigmoid::~Sigmoid(){Reset();}
Sigmoid& Sigmoid::operator=(const Sigmoid& function)
{
	ActivationFunction::operator=(function);
	exponent = function.exponent;
	return *this;
}
void Sigmoid::Reset()
{
	ActivationFunction::Reset();
	Initialize();
}
double Sigmoid::Evaluate(const double& argument) const{return (1.0/(1.0 + exp(-exponent*argument)));}
double Sigmoid::Differentiate(const double& argument) const
{
	double f = Evaluate(argument);
	return (exponent*f*f*exp(-exponent*argument));
}
void Sigmoid::Exponent(const double& value){exponent = value;}
double Sigmoid::Exponent() const{return exponent;}
ActivationFunction* Sigmoid::Clone() const{return new Sigmoid(*this);}
void Sigmoid::Write(FILE* file) const{fprintf(file,"sigmoid\t\t%e\n",exponent);}
bool Sigmoid::Read(FILE* file)
{
	char line[1024];
	if(!EZ::IO::ReadLine(line,1024,file))		return false;
	sscanf(line,"sigmoid\t\t%lf\n",&exponent);
	return true;
}
void Sigmoid::Initialize()
{
	ActivationFunction::Initialize();
	exponent = 1.0;
}

ReLU::ReLU(){Initialize();}
ReLU::ReLU(const ReLU& function) : ActivationFunction(function){*this = function;}
ReLU::~ReLU(){Reset();}
ReLU& ReLU::operator=(const ReLU& function)
{
	ActivationFunction::operator=(function);
	leak_coefficient = function.leak_coefficient;
	return *this;
}
void ReLU::Reset()
{
	ActivationFunction::Reset();
	Initialize();
}
double ReLU::Evaluate(const double& argument) const
{
	if(argument < 0.0)		return (leak_coefficient*argument);
	return argument;
}
double ReLU::Differentiate(const double& argument) const
{
	if(argument < 0.0)		return leak_coefficient;
	return 1.0;
}
void ReLU::LeakCoefficient(const double& value){leak_coefficient = value;}
double ReLU::LeakCoefficient() const{return leak_coefficient;}
ActivationFunction* ReLU::Clone() const{return new ReLU(*this);}
void ReLU::Write(FILE* file) const{fprintf(file,"relu\t\t%e\n",leak_coefficient);}
bool ReLU::Read(FILE* file)
{
	char line[1024];
	if(!EZ::IO::ReadLine(line,1024,file))		return false;
	sscanf(line,"relu\t\t%lf\n",&leak_coefficient);
	return true;
}
void ReLU::Initialize()
{
	ActivationFunction::Initialize();
	leak_coefficient = 0.0;
}

