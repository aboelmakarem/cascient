// Activation.h
// Ahmed M. Hussein (amhussein4@gmail.com)
// 08/17/2020

#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include "stdio.h"

class ActivationFunction
{
public:
	ActivationFunction();
	ActivationFunction(const ActivationFunction& function);
	virtual ~ActivationFunction();
	virtual ActivationFunction& operator=(const ActivationFunction& function);
	virtual void Reset();
	double operator()(const double& argument) const;
	virtual double Evaluate(const double& argument) const = 0;
	virtual double Differentiate(const double& argument) const = 0;
	virtual ActivationFunction* Clone() const = 0;
	virtual void Write(FILE* file) const = 0;
	virtual bool Read(FILE* file) = 0;
	static ActivationFunction* ReadAndCreate(FILE* file);

private:

protected:
	virtual void Initialize();
};

class Identity : public ActivationFunction
{
public:
	Identity();
	Identity(const Identity& function);
	~Identity();
	Identity& operator=(const Identity& function);
	void Reset();
	double Evaluate(const double& argument) const;
	double Differentiate(const double& argument) const;
	ActivationFunction* Clone() const;
	void Write(FILE* file) const;
	bool Read(FILE* file);

private:
	void Initialize();
};

class Sigmoid : public ActivationFunction
{
public:
	Sigmoid();
	Sigmoid(const Sigmoid& function);
	~Sigmoid();
	Sigmoid& operator=(const Sigmoid& function);
	void Reset();
	double Evaluate(const double& argument) const;
	double Differentiate(const double& argument) const;
	void Exponent(const double& value);
	double Exponent() const;
	ActivationFunction* Clone() const;
	void Write(FILE* file) const;
	bool Read(FILE* file);

private:
	void Initialize();
	double exponent;
};

class ReLU : public ActivationFunction
{
public:
	ReLU();
	ReLU(const ReLU& function);
	~ReLU();
	ReLU& operator=(const ReLU& function);
	void Reset();
	double Evaluate(const double& argument) const;
	double Differentiate(const double& argument) const;
	void LeakCoefficient(const double& value);
	double LeakCoefficient() const;
	ActivationFunction* Clone() const;
	void Write(FILE* file) const;
	bool Read(FILE* file);

private:
	void Initialize();
	double leak_coefficient;
};

#endif

