#pragma once

#include "matrix.h"
#include <list>
#include <memory>


namespace nn{
	using nnLayer = Matrix<double>::Layer;
	
	template<size_t N>
	class ReLU: nnLayer{	
	
	private:
		Matrix<double>* res_ptr=nullptr;
		Matrix<double>* input_ptr=nullptr;
		Matrix<bool> mask;

		virtual void change_res_ptr(Matrix<double>* ptr) {
			res_ptr = ptr;
		}

	public:
		ReLU() = default;

		Matrix<double> forward(Matrix<double>& x) {
			input_ptr = &x;
			mask(x.size(), 0, nullptr);
			
			Matrix<double> result = x;
			for(size_t i = 0; i < result.num_rows(); ++i){
				for(size_t j = 0; j < result.num_columns(); ++j){
					result[i][j] = std::max(result[i][j], 0.0);
					mask[i][j] = (result[i][j] > 0.0);
				}
			}
			result.layer_ptr = shared_from_this();
			return result;
		}

		void backward(const Matrix<double>& grad_other){
			auto& grad_current = *(res_ptr->grad_ptr);
			grad_current += grad_other;
			Matrix<double> grad_push(grad_other);
			for(size_t i = 0; i < grad_push.num_rows(); ++i){
				for(size_t j = 0; j < grad_push.num_columns(); ++j){
					grad_push[i][j] *= mask[i][j];
				}
			}
			input_ptr->backward(grad_push);
		}
	};
	


	class MSELoss: nnLayer{
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
		Matrix<double>* real_ptr=nullptr;
		Matrix<double>* pred_ptr=nullptr;
		Matrix<double>* res_ptr=nullptr;

		void change_res_ptr(Matrix<double>* ptr){
			res_ptr = ptr;
		}

	public:
	Matrix<double> forward(Matrix<double>& real, Matrix<double>& pred){
			pred_ptr = &pred;
			real_ptr = &real;

			if(real.size() != pred.size()){
				throw BadShape("Wrong sizes. MSELoss, forward");
			}

			Matrix<double> res(1, 1, 0, nullptr);
			res->layer_ptr = shared_from_this();

			for(size_t i = 0; i < real.num_rows(); ++i){
				for(size_t j = 0; j < real.num_columns(); ++j){
					res[0][0] += std::pow(y[i][j] - pred[i][j], 2);
				}
			}
			return res;
		}

		void backward(const Matrix<double>& _not_used=Matrix<double>()){
			//I don't care about grad_current

			Matrix<double> grad_push(*pred_ptr);
			grad_push -= *real_ptr;
			grad_push *= 2;

			pred_ptr->backward(grad_push);
		}
	};

	template<size_t In, size_t Out>
	class Linear: public nnLayer{
	/*
		x: N x (In + 1)
		w: (In + 1) x Out
		

		let after it will be f(y) and we have grad_y where y = x*w
		so we need to find dy/dw

		let's look at only one vector => 
		x: 1 x M = (x_1, ... , x_M)
		y = xw: 1 x C = (y_1, ... , y_C)
		grad_y: 1 x C = (df(y)/dy_1, ... , df(y)/dy_C)

			(w_{1 1}, ..., w_{1 C})
		w = 			...			=> y = (x_1 * w_{1 1} + ... + x_M * w_{M 1}, ... , x_1 * w_{1 C} + ... + x_M * w_{M C})
			(w_{M 1}, ..., w_{M C})

		df(y)/dw_{i j} = df(y)/dy_j * dy_j/dw_{i j} = df(y)/dy_j * x_i

					(df(y)/dy_1 * x_1 , ... , df(y)/dy_C * x_1)		(x_1)
		grad_w = 						...						= 	...	 * 	(df(y)/dy_1, ... , df(y)/dy_C) = x^T * grad_y
					(df(y)/dy_1 * x_M , ... , df(y)/dy_C * x_M)		(x_M)

	*/
	private:
		Matrix<double> result;
		Matrix<double> grad(In + 1, Out, 0.0);
		Matrix<double> w;

	public:
		Linear(): w(Matrix<double>::random(In + 1, Out)){}

		Matrix<double> forward(const Matrix<double>& x) override{
			if(x.num_columns() != In + 1){
				throw BadShape("x in forward of linear has wrong shape");
			}
			result = x * w;
			return result;
		}
		
		Matrix<double> backward(const Matrix<double>& other_grad) override{
			grad += result.transpose() * other_grad;
			return grad;
		}

		void zero_grad(){
			graz.make_zero();
		}

		void make_step(double step){
			grad *= step;
			w -= grad;
		}

		~Linear() = default;
	};

	/*
	template<typename... Layers>
	class Sequantial{
	private:
		std::list<std::unique_ptr<nnLayer>> seq;

		auto add_ones(const Matrix<double>& x_fresh) const{
			Matrix<double> res(x_fresh);
			for(size_t i = 0; i < res.num_rows(); ++i){
				res[i].push_back(1.0);
			}
		}
		template<typename... LLayers>
		void PushBackMany(LLayers&&... layers){}

		template<typename Head, typename... LLayers>
		void PushBackMany(Head&& head, LLayers&&... layers){
			seq.push_back(new Head(std::forward<Head>(head)));
			PushBackMany(std::forward<LLayers>(layers)...);
		}

	public:
		static const size_t length = sizeof...(Layers);
		Sequantial(): Sequantial(Layers()...){}

		template<typename... LLayers>
		Sequantial(LLayers&&... layers){
			PushBackMany(std::forward<LLayers>(layers)...);
		}

		auto forward(const Matrix<double>& x_fresh) {
			auto x = x_fresh;
			for(auto& layer_ptr: seq){
				x = layer_ptr->forward(x);
			}
			return x;
		}

		void backward(Matrix<double>& grad_other){
			seq.back()->backward(grad_other);
		}

		~Sequantial(){}
	};
	*/

}

