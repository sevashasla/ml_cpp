#pragma once

#include "basic_layer.hpp"

namespace nn{
template<typename Field>
class Multiplier: public Layer<Field>{

/*
	[A] = M x N
	[B] = N x K
	[C] = M x K, C = A * B
				(df/dC_{1 1}, ... , df/dC_{1 k})
	grad_C = 				...
				(df/dC_{M 1}, ... , df/dC_{m k})
	df/dA_{p q} = sum(df/dC_{i j} * dC_{i j} / dA_{p q}),
	but C_{i j} = sum(A_{i k} * B_{k j})
	df/dA_{1 1} = sum(df/dC_{1 i} * dC_{1 i}/ dA_{1 1}) = sum(df/dC_{1 i} * B_{1 i})
	so 
	grad_A = grad_C * B^T
	grad_B = A^T * grad_C
*/

private:
	Tensor<Field>* left_ptr;
	Tensor<Field>* right_ptr;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:
	Multiplier(){}

	Tensor<Field> forward(Tensor<Field>& left, Tensor<Field>& right) override{
		left_ptr = &left;
		right_ptr = &right;

		Tensor<Field> result(
			static_cast<Matrix<Field>&>(left) * static_cast<Matrix<Field>&>(right), 
			SharedBase::shared_from_this()
		);

		return result;
	}

	void backward_(const Matrix<Field>& grad) override{
		left_ptr->backward(grad * right_ptr->transpose());
		right_ptr->backward(left_ptr->transpose() * grad);
	}
	
	void zero_grad_() override {
		left_ptr->zero_grad();
		right_ptr->zero_grad();
	}

	void make_step_(Field step) override{
		left_ptr->make_step(step);
		right_ptr->make_step(step);
	}

	void break_graph_() override {
		left_ptr->break_graph();
		right_ptr->break_graph();
	}

	~Multiplier() = default;
};


template<typename Field>
class Adder: public Layer<Field>{
/*
	A + B = C
	y = f(C)
			(dy/dC_{1 1}, ... , dy/dC_{1 N})
	grad_y = 			...
			(dy/dC_{M 1}, ... , dy/dC_{M N})
	

			(dy/dA_{1 1}, ... , dy/dA_{1 N})	(dy/dC_{1 1}, ... , dy/dC_{1 N})
	grad_A = 			...					=				...
			(dy/dA_{M 1}, ... , dy/dA_{M N})	(dy/dC_{1 1}, ... , dy/dC_{1 N})
				
	similarly for grad_B
*/
private:
	// It is more approptiate to use "usual" pointer
	// Because forward get reference to object
	Tensor<Field>* left_ptr=nullptr;
	Tensor<Field>* right_ptr=nullptr;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:
	Adder(){}

	// non-copyable
	Adder(const Adder&) = delete;
	Adder& operator=(const Adder&) = delete;

	// non-movable
	Adder(Adder&&) = delete;
	Adder& operator=(Adder&&) = delete;	


	Tensor<Field> forward(Tensor<Field>& left, Tensor<Field>& right) override{
		left_ptr = &left;
		right_ptr = &right;
		Tensor<Field> result(
			static_cast<Matrix<Field>&>(left) + static_cast<Matrix<Field>&>(right), 
			SharedBase::shared_from_this()
		);
		return result;
	}

	void backward_(const Matrix<Field>& grad) override{
		left_ptr->backward(grad);
		right_ptr->backward(grad);
	}

	void zero_grad_() override{
		left_ptr->zero_grad();
		right_ptr->zero_grad();
	}

	void make_step_(Field step) override{
		left_ptr->make_step(step);
		right_ptr->make_step(step);
	}

	void break_graph_() override {
		left_ptr->break_graph();
		right_ptr->break_graph();
	}

	~Adder() = default;
};


template<typename Field>
class Subtractor: public Layer<Field>{
private:
	Tensor<Field>* left_ptr;
	Tensor<Field>* right_ptr;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:

	Tensor<Field> forward(Tensor<Field>& left, Tensor<Field>& right) override{
		left_ptr = &left;
		right_ptr = &right;

		Tensor<Field> result(
			static_cast<Matrix<Field>&>(left) - static_cast<Matrix<Field>&>(right), 
			SharedBase::shared_from_this()
		);
		
		return result;
	}

	void backward_(const Matrix<Field>& grad) override{
		left_ptr->backward(grad);

		right_ptr->backward(grad * (-1));
	}

	void zero_grad_() override{
		left_ptr->zero_grad();
		right_ptr->zero_grad();
	}

	void make_step_(Field step) override{
		left_ptr->make_step(step);
		right_ptr->make_step(step);
	}

	void break_graph_() override {
		left_ptr->break_graph();
		right_ptr->break_graph();
	}
};


template<typename Field>
class Transposer: public Layer<Field>{
private:
	Tensor<Field>* input_ptr;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:
	Transposer() = default;

	Tensor<Field> forward(Tensor<Field>& input) override {
		input_ptr = &input;
		Tensor<Field> result(
			static_cast<Matrix<Field>&>(input)->transpose(),
			SharedBase::shared_from_this()
		);
		return result;
	}


	void backward_(const Matrix<Field>& grad) override {
		input_ptr->backward(grad.transpose());
	}

	void zero_grad_() override {
		input_ptr->zero_grad();
	}

	void make_step_(Field step) override {
		input_ptr->make_step(step);
	}

	void break_graph_() override {
		input_ptr->break_graph();
	}
};


} // end of namespace nn
