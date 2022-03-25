#pragma once

#include <basic_layer.hpp>

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
	Matrix<Field>* left_ptr;
	Matrix<Field>* right_ptr;

public:
	Multiplier(){}

	static Matrix<Field> matmul(const Matrix<Field>& left, const Matrix<Field>& right){
	/*
		c[i][j] = sum_k_(a[i][k] * b[k][j]);
		(M x N) * (N x K) -> (M x K)
	*/

		size_t M = left.num_rows();
		size_t N = left.num_columns();
		size_t K = right.num_columns();


		Layer<Field> result(M, K, 0);
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < K; ++j){
				for(size_t k = 0; k < N; ++k){
					result[i][j] += left[i][k] * right[k][j];
				}
			}
		}

		return result;
	}

	static Matrix<double> mulscalar(double scalar, const Matrix<double>& matrix){
		Matrix<double> result = matrix;
		for(size_t i = 0; i < result.num_rows(); ++i){
			for(size_t j = 0; j < result.num_columns(); ++j){
				result[i][j] *= scalar;
			}
		}
		return result;
	}

	Matrix<double> forward(Matrix<double>& left, Matrix<double>& right) override{
		if(left.num_columns() != right.num_rows()){
			throw BadShape("sizes must be m x n * n x k. Multiplier, forward");
		}

		left_ptr = &left;
		right_ptr = &right;
		Matrix<double> result = matmul(left, right);
		result.layer_ptr = shared_from_this();
		return result;
	}

	void backward() override{
		auto& grad_current = *(res_ptr->grad_ptr);

		left_ptr->backward(matmul(grad_current, right_ptr->transpose()));
		right_ptr->backward(matmul(left_ptr->transpose(), grad_current));
	}
	
	void zero_grad(){
		left_ptr->zero_grad();
		right_ptr->zero_grad();
	}

	void make_step(double step) override{
		left_ptr->make_step(step);
		right_ptr->make_step(step);
	}

	void break_graph(){
		left_ptr->break_graph();
		right_ptr->break_graph();
	}

	~Multiplier() = default;
};



// class Adder: public Matrix<double>::Layer{
// /*
// 	A + B = C
// 	y = f(C)
// 			(dy/dC_{1 1}, ... , dy/dC_{1 N})
// 	grad_y = 			...
// 			(dy/dC_{M 1}, ... , dy/dC_{M N})
	

// 			(dy/dA_{1 1}, ... , dy/dA_{1 N})	(dy/dC_{1 1}, ... , dy/dC_{1 N})
// 	grad_A = 			...					=				...
// 			(dy/dA_{M 1}, ... , dy/dA_{M N})	(dy/dC_{1 1}, ... , dy/dC_{1 N})
				
// 	similarly for grad_B
// */
// private:
// 	Matrix<double>* left_ptr=nullptr;
// 	Matrix<double>* right_ptr=nullptr;

// 	static Matrix<double> add(Matrix<double>& left, Matrix<double>& right){
// 		Matrix<double> result = left;
// 		for(size_t i = 0; i < left.num_rows(); ++i){
// 			for(size_t j = 0; j < left.num_columns(); ++j){
// 				result[i][j] += right[i][j];
// 			}
// 		}
// 		return result;		
// 	}

// public:
// 	Adder(const Adder&) = delete;
// 	Adder& operator=(const Adder&) = delete;

// 	Adder(){}

// 	Matrix<double> forward(Matrix<double>& left, Matrix<double>& right) override{
// 		if(left.size() != right.size()){
// 			throw BadShape("Wrong shapes of matrices. Adder, forward");
// 		}

// 		left_ptr = &left;
// 		right_ptr = &right;
		
// 		Matrix<double> result = add(left, right);
// 		result.layer_ptr = shared_from_this();

// 		return result;
// 	}

// 	void backward() override{
// 		auto& grad_current = *(res_ptr->grad_ptr);

// 		left_ptr->backward(grad_current);
// 		right_ptr->backward(grad_current);
// 	}

// 	void zero_grad(){
// 		left_ptr->zero_grad();
// 		right_ptr->zero_grad();
// 	}

// 	void make_step(double step) override{
// 		left_ptr->make_step(step);
// 		right_ptr->make_step(step);
// 	}

// 	void break_graph(){
// 		left_ptr->break_graph();
// 		right_ptr->break_graph();
// 	}

// 	~Adder() = default;
// };


// class Subtractor: public Matrix<double>::Layer{
// private:
// 	Matrix<double>* left_ptr;
// 	Matrix<double>* right_ptr;

// 	static Matrix<double> subtract(const Matrix<double>& left, const Matrix<double>& right){
// 		Matrix<double> result(left);
// 		for(size_t i = 0; i < left.num_rows(); ++i){
// 			for(size_t j = 0; j < left.num_columns(); ++j){
// 				result[i][j] -= right[i][j];
// 			}
// 		}
// 		return result;
// 	}

// public:

// 	Matrix<double> forward(Matrix<double>& left, Matrix<double>& right) override{
// 		if(left.size() != right.size()){
// 			throw BadShape("Wrong shapes of matrices. Subtractor, forward");
// 		}

// 		left_ptr = &left;
// 		right_ptr = &right;

// 		Matrix<double> result = subtract(left, right);
// 		result.layer_ptr = shared_from_this();

// 		return result;
// 	}

// 	void backward() override{
// 		auto& grad_current = *(res_ptr->grad_ptr);
// 		left_ptr->backward(grad_current);
// 		right_ptr->backward(Multiplier::mulscalar(-1, grad_current));
// 	}

// 	void zero_grad() override{
// 		left_ptr->zero_grad();
// 		right_ptr->zero_grad();
// 	}

// 	void make_step(double step) override{
// 		left_ptr->make_step(step);
// 		right_ptr->make_step(step);
// 	}

// 	void break_graph() override{
// 		left_ptr->break_graph();
// 		right_ptr->break_graph();
// 	}
// };


// class Transposer: public Matrix<double>::Layer{
// private:
// 	Matrix<double>* res_ptr;
// 	Matrix<double>* input_ptr;

// public:
// 	Transposer() = default;

// 	static Matrix<double> transpose(const Matrix<double>& input){
// 		Matrix<double> result(input.num_columns(), input.num_rows());
// 		for(size_t i = 0; i < input.num_columns(); ++i){
// 			for(size_t j = 0; j < input.num_rows(); ++j){
// 				result[i][j] = input[j][i];
// 			}
// 		}
// 		return result;
// 	}

// 	Matrix<double> forward(Matrix<double>& input) override {
// 		input_ptr = &input;
// 		Matrix<double> result = transpose(input);
// 		result.layer_ptr = shared_from_this();
// 		return result;
// 	}


// 	void backward() override {
// 		auto& grad_current = *(res_ptr->grad_ptr);
// 		input_ptr->backward(transpose(grad_current));
// 	}

// 	void zero_grad() override {
// 		input_ptr->zero_grad();
// 	}

// 	void make_step(double step) override {
// 		input_ptr->make_step(step);
// 	}

// 	void break_graph() override {
// 		input_ptr->break_graph();
// 	}
// };


} // end of namespace nn
