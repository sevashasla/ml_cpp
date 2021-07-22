#pragma once

#include "matrix.h"


class Layer{
public:
	virtual void backward(const Matrix<double>&) = 0;
	virtual Matrix<double>& forward(Matrix<double>&, Matrix<double>&) = 0;
	virtual ~Layer() = default;
};


class Multiplier: public Layer{
/*
	A: M x N
	B: N x K
	C: M x K = A * B
	
	C_i = (A_{i 1} * B{1 1} + ... + A_{i N} * B_{N 1}, ... , A_{i 1} * B{1 K} + ... + A_{i N} * B{N K})

	df/dA_{1 1} = 1/K * (df/dC_{1 1} * B_{1 1} + ... + df/dC_{1 K} * B_{K 1})
	df/dB_{1 1} = 1/M * (df/dC_{1 1} * A_{1 1} + ... + df/dC_{M 1} * A_{M 1})

			(df/dC_{1 1}, ... , df/dC_{1 K})
	grad_f = 				...
			(df/dC_{M 1}, ... , df/dC_{M K})
	
	grad_A = 1/K * grad_f * B^T
	grad_B = 1/M * A^T * grad_f
*/

private:
	Matrix<double>* left_ptr;
	Matrix<double>* right_ptr;
	Matrix<double>* res_ptr;

	Matrix<double> matmul(const Matrix<double>& left, const Matrix<double>& right){
	/*
		c[i][j] = sum_k_(a[i][k] * b[k][j]);
		(M x N) * (N x K) -> (M x K)
	*/

		size_t M = left.num_rows();
		size_t N = left.num_columns();
		size_t K = right.num_columns();

		Matrix<double> result(M, K, 0, this);
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < K; ++j){
				for(size_t k = 0; k < N; ++k){
					result[i][j] += left[i][k] * right[k][j];
				}
			}
		}

		return result;
	}

	void mulscalar_helper(double scalar, Matrix<double>& matrix) const{
		for(size_t i = 0; i < matrix.num_rows(); ++i){
			for(size_t j = 0; j < matrix.num_columns(); ++j){
				matrix[i][j] *= scalar;
			}
		}
	}

	Matrix<double> mulscalar(double scalar, const Matrix<double>& matrix){
		Matrix<double> result = matrix;
		mulscalar_helper(scalar, result);
		return result;
	}

	Matrix<double> mulscalar(double scalar, Matrix<double>&& matrix){
		Matrix<double> result = std::move(matrix);
		mulscalar_helper(scalar, result);
		return result;
	}


public:
	Multiplier(){
		res_ptr = new Matrix<double>(0, 0);
		res_ptr->layer = this;
	}

	Multiplier(const Multiplier& other): Multiplier(){
		*res_ptr = *other.res_ptr;
		res_ptr->layer = this;
		left_ptr = other.left_ptr;
		right_ptr = other.right_ptr;
	}

	Multiplier(Multiplier&& other){
		res_ptr = other.res_ptr;
		res_ptr->layer = this;
		left_ptr = other.left_ptr;
		right_ptr = other.right_ptr;
	}

	Matrix<double>& forward(Matrix<double>& left, Matrix<double>& right) override{
		if(left.num_columns() != right.num_rows()){
			delete res_ptr;
			throw BadShape("sizes must be m x n * n x k. Multiplier, forward");
		}

		*res_ptr = matmul(left, right);
		return *res_ptr;
	}


	void backward(const Matrix<double>& grad_other) override{
		*(res_ptr->grad_ptr) += grad_other;

		//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!grad_other or something other?
		left_ptr->backward(mulscalar(1.0 / right_ptr->num_columns(), matmul(grad_other, right_ptr->transpose())));
		right_ptr->backward(mulscalar(1.0 / left_ptr->num_rows(), matmul(left_ptr->transpose(), grad_other)));
	}

	~Multiplier(){
		delete res_ptr;
	}
};


class Adder: public Layer{
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
	Matrix<double>* left_ptr;
	Matrix<double>* right_ptr;
	Matrix<double>* res_ptr;
	

public:
	Adder(){
		res_ptr = new Matrix<double>(0, 0);
	}

	Matrix<double>& forward(Matrix<double>& left, Matrix<double>& right) override{
		if(left.size() != right.size()){
			delete res_ptr;

			throw BadShape("left and right must have the same sizes. Adder, forward");
		}

		this->left_ptr = &left;
		this->right_ptr = &right;

		*res_ptr = left;
		res_ptr->layer = this;

		for(size_t i = 0; i < left.num_rows(); ++i){
			for(size_t j = 0; j < left.num_columns(); ++j){
				(*res_ptr)[i][j] += (left[i][j] + right[i][j]);
			}
		}

		return *res_ptr;
	}

	void backward(const Matrix<double>& grad_other) override{
		*(res_ptr->grad_ptr) += grad_other;
		left_ptr->backward(grad_other);
		right_ptr->backward(grad_other);
	}

	~Adder(){
		delete res_ptr;
	}
};
