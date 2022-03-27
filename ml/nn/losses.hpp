#pragma once

#include "../tensor/tensor.hpp"
#include "../tensor/basic_layer.hpp"

namespace ml::nn::losses {

template<typename Field>
class MSELoss: public Layer<Field> {
/*
	real: N x C
	pred: N x C
	let's look at only one
	real: 1 x C = (real_1, ... , real_C)
	pred: 1 x C = (real_pred_1, ... , real_pred_C)

	f = (real_1 - real_pred_1) ^ 2 + ... + (real_C - real_pred_C) ^ 2
	df/dreal_pred_i = 2 * (real_pred_i - real_i)
	grad_real_pred = 2 * (pred - real)
*/

private:
	Tensor<Field>* real_ptr=nullptr;
	Tensor<Field>* pred_ptr=nullptr;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:
	MSELoss() = default;

	Tensor<Field> forward(Tensor<Field>& real, Tensor<Field>& pred) override{
		if(real.size() != pred.size()){
			throw BadShape("Wrong sizes. MSELoss, forward");
		}

		pred_ptr = &pred;
		real_ptr = &real;

		Tensor<Field> result(Matrix<Field>(1, 1, 0), SharedBase::shared_from_this());

		for(size_t i = 0; i < real.num_rows(); ++i){
			for(size_t j = 0; j < real.num_columns(); ++j){
				result[0][0] += std::pow(real[i][j] - pred[i][j], 2.0);
			}
		}
		return result;
	}

	void backward_(const Matrix<Field>&) override{
		Matrix<Field> grad_push(*pred_ptr);
		grad_push -= *real_ptr;
		grad_push *= 2;
		grad_push /= pred_ptr->num_rows();

		pred_ptr->backward(grad_push);
	}

	void make_step_(Field step){
		pred_ptr->make_step(step);
	}

	void zero_grad_(){
		pred_ptr->zero_grad();
	}

	void break_graph_(){
		pred_ptr->break_graph();
		real_ptr->break_graph();
	}

	~MSELoss() = default;
};


template<typename Field, size_t Classes>
class CrossEntropyLoss: public Layer<Field>{
/*
	x: 1 x C
	y: 1 x C
	p: 1 x C

	dlog(p_i)/dx_j = d log(e^x_i / sum(e^x_k)) / dx_j = 
	= dx_i/dx_j - d(log(sum(e^x_k)))/dx_j = 
	= dx_i/dx_j - 1 / sum(e^x_k) * d(sum(e^x_k))/dx_j = 
	= (i == j ? 1 : 0) - p_j = (i == j ? 1 - p_j : -p_j)
	
	df/dx_i = -(y_1 * (1 - p_1) + y_2 * -p_1 ... + y_C * -p_1) =(!!!) p_1 - y_1
	so grad_x = p - y
*/

private:
	Tensor<Field>* real_ptr;
	Tensor<Field>* pred_ptr;
	Matrix<Field> logits;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:

	CrossEntropyLoss() = default;

	Tensor<Field> forward(Tensor<Field>& real, Tensor<Field>& pred) override{
		if(real.size() != pred.size()){
			throw BadShape("Wrong sizes of matrices. CrossEntropyLoss, forward");
		}

		if(real.num_columns() != Classes){
			throw BadShape("Wrong size of matrix, should be N x Classes. CrossEntropyLoss, forward");
		}

		real_ptr = &real;
		pred_ptr = &pred;

		logits = Matrix<Field>(pred.size(), 0.0);
		for(size_t num = 0; num < pred.num_rows(); ++num){
			Field _sum = 0.0;
			for(size_t i = 0; i < Classes; ++i){
				_sum += std::exp(pred[num][i]);
			}

			for(size_t i = 0; i < Classes; ++i){
				logits[num][i] = std::exp(pred[num][i]) / _sum;
			}
		}

		Tensor<Field> result(Matrix<Field>(1, 1, 0), SharedBase::shared_from_this());

		for(size_t num = 0; num < pred.num_rows(); ++num){
			for(size_t i = 0; i < Classes; ++i){
				result[0][0] -= std::log(logits[num][i]) * real[num][i];
			}
		}
		return result;
	}

	void backward_(const Matrix<Field>&){
		Matrix<Field> grad_push(logits);
		grad_push -= *real_ptr;
		grad_push /= pred_ptr->num_rows();
		pred_ptr->backward(grad_push);
	}

	void make_step_(Field step){
		pred_ptr->make_step(step);
	}

	void zero_grad_(){
		pred_ptr->zero_grad();
	}

	void break_graph_(){
		pred_ptr->break_graph();
	}
};

} // end of ml::nn::losses