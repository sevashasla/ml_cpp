/*
	let f = f(y_1, ... , y_k)
	y_1 = g_1(x), ... , y_k = g_k(x)
	d(f o g)(a) = df(g(a)) o dg(a) = D_f(g(a)) * D_g(a) - two matrices
	so df/dx = sum(df/dy_i * dy_i/dx)
*/


//TODO
/*
	1) Do I need grad_other in backward in Layer?
	2) I can make input_ptr, left_ptr, right_ptr in Layer
	so I don't need to copy-paste code

*/


#pragma once

#include "matrix.h"
#include <list>
#include <memory>


template<typename...>
void f() = delete;

namespace nn{
	using nnLayer = Matrix<double>::Layer;
	

	class ReLU: public nnLayer{
	private:
		Matrix<double>* input_ptr=nullptr;
		Matrix<bool> mask;

	public:
		ReLU() = default;

		Matrix<double> forward(Matrix<double>& input) override{
			input_ptr = &input;
			mask = Matrix<bool>(input.size(), 0, nullptr);
			
			Matrix<double> result = input;
			for(size_t i = 0; i < result.num_rows(); ++i){
				for(size_t j = 0; j < result.num_columns(); ++j){
					result[i][j] = std::max(result[i][j], 0.0);
					mask[i][j] = (result[i][j] > 0.0);
				}
			}
			result.layer_ptr = shared_from_this();
			return result;
		}

		void backward() override{
			auto& grad_current = *(res_ptr->grad_ptr);
			
			Matrix<double> grad_push(grad_current);
			for(size_t i = 0; i < grad_push.num_rows(); ++i){
				for(size_t j = 0; j < grad_push.num_columns(); ++j){
					grad_push[i][j] *= mask[i][j];
				}
			}
			input_ptr->backward(grad_push);
		}

		void make_step(double step){
			input_ptr->make_step(step);
		}

		void zero_grad(){
			input_ptr->zero_grad();
		}

		void break_graph() override{
			input_ptr->break_graph();
		}
	};
	

	class LeakyReLU: public nnLayer{
		private:
		Matrix<double>* input_ptr=nullptr;
		Matrix<double> mask;
		double alpha=0.0;

	public:
		LeakyReLU(double alpha=1e-2): alpha(alpha){}

		Matrix<double> forward(Matrix<double>& input) override{
			input_ptr = &input;
			mask = Matrix<double>(input.size(), 0, nullptr);
			
			Matrix<double> result = input;
			for(size_t i = 0; i < result.num_rows(); ++i){
				for(size_t j = 0; j < result.num_columns(); ++j){
					result[i][j] = std::max(result[i][j], alpha * result[i][j]);
					mask[i][j] = (result[i][j] > 0.0 ? 1 : alpha);
				}
			}
			result.layer_ptr = shared_from_this();
			return result;
		}

		void backward() override{
			auto& grad_current = *(res_ptr->grad_ptr);
			
			Matrix<double> grad_push(grad_current);
			for(size_t i = 0; i < grad_push.num_rows(); ++i){
				for(size_t j = 0; j < grad_push.num_columns(); ++j){
					grad_push[i][j] *= mask[i][j];
				}
			}
			input_ptr->backward(grad_push);
		}

		void make_step(double step){
			input_ptr->make_step(step);
		}

		void zero_grad(){
			input_ptr->zero_grad();
		}

		void break_graph() override{
			input_ptr->break_graph();
		}
	};


	class Sigmoid: public nnLayer{
	/*
		y = 1/(1 + e^-x)
		df/dx = d(1/(1 + e^-x))/dx = -1/(1 + e^-x)^2 * -e^-x = 
		= e^-x/(1 + e^-x)^2 = y(1 - y)
		I have df/dy => df/dx = df/dy * dy/dx
	*/

	private:
		Matrix<double>* input_ptr;

	public:

		Matrix<double> forward(Matrix<double>& input){
			input_ptr = &input;

			Matrix<double> result(input.size(), 0.0, nullptr);
			for(size_t num = 0; num < input.num_rows(); ++num){
				for(size_t i = 0; i < input.num_columns(); ++i){
					result[num][i] = 1.0 / (1.0 + std::exp(-input[num][i]));
				}
			}
			result.layer_ptr = shared_from_this();
			return result;
		}


		void backward(){
			auto& grad_current = *(res_ptr->grad_ptr);
			Matrix<double> grad_push(grad_current.size(), 0.0);
			for(size_t num = 0; num < grad_current.num_rows(); ++num){
				for(size_t i = 0; i < grad_current.num_columns(); ++i){
					auto& res_curr = (*res_ptr)[num][i];
					grad_push[num][i] = grad_current[num][i] * (res_curr) * (1 - res_curr);
				}
			}
			input_ptr->backward(grad_push);
		}

		void zero_grad(){
			input_ptr->zero_grad();
		}

		void make_step(double step){
			input_ptr->make_step(step);
		}

		void break_graph(){
			input_ptr->break_graph();
		}
	};


	class BatchNorm: public nnLayer{
	/*
		let's look at x: N x 1 = (x_1 , ... , x_N)^T
		y_i = (x_i - mean) / std

		grad_y = (df/dy_1, ... , df/dy_N)
		grad_x = (df/dx_1, ... , df/dx_N)

		df/dx_j = sum(df/dy_k * dy_k/dx_j)

		dy_i / dx_j = d[(x_i - mean) / std]/dx_j
		d(x_j)/dx_i = (i == j)
		d(mean)/dx_j = 1/N
		d(D)/dx_j = 1/N * d(sum[(x_k - mean)^2])/dx_j = 
		= 1/N * sum[2 * (x_k - mean) * (k == j - 1/N)] = 
		= 2/N * sum[(x_k - mean) * -1/N] + sum[(x_k - mean) * (k == j)] = 
		= 2/N * (x_j - mean)
		d(std)/dx_j = d(sqrt(D))/dx_j = 1/(2 sqrt(D)) * 2/N * (x_j - mean) = 
		= 1/(N * std) * (x_j - mean)

		dy_i/dx_j = [(i == j - 1/N)(std) - (x_i - mean)(1/(N * std) * (x_j - mean))]/[std^2]
		
	*/
	private:
		Matrix<double>* input_ptr;
		Matrix<double> __mean;
		Matrix<double> __std;


	public:
		BatchNorm() = default;

		static Matrix<double> mean(const Matrix<double>& input){
			size_t num_out = input.num_columns();
			size_t N = input.num_rows();

			Matrix<double> result(1, num_out, 0.0, nullptr);
			for(size_t num = 0; num < N; ++num){
				for(size_t i = 0; i < num_out; ++i){
					result[0][i] += input[num][i];
				}
			}
			result /= N;
			return result;
		}

		static Matrix<double> std(const Matrix<double>& input){
			size_t num_out = input.num_columns();
			size_t N = input.num_rows();

			Matrix<double> result(1, num_out, 0.0, nullptr);
			Matrix<double> __mean = mean(input);

			for(size_t num = 0; num < N; ++num){
				for(size_t i = 0; i < num_out; ++i){
					result[0][i] += std::pow(input[num][i] - __mean[0][i], 2);
				}
			}
			result /= N;
			for(size_t i = 0; i < num_out; ++i){
				result[0][i] = std::sqrt(result[0][i]);
			}
			return result;
		}

		Matrix<double> forward(Matrix<double>& input){
			input_ptr = &input;

			Matrix<double> result(input);

			__mean = mean(input);
			__std = std(input);

			result.layer_ptr = shared_from_this();

			for(size_t num = 0; num < input.num_rows(); ++num){
				for(size_t i = 0; i < input.num_columns(); ++i){
					result[num][i] -= __mean[0][i];
					result[num][i] /= __std[0][i];
				}
			}

			return result;
		}

		void backward(){
			auto& grad_current = *(res_ptr->grad_ptr);
			Matrix<double> grad_push(grad_current.size(), 0.0);
			size_t num_out = input_ptr->num_columns();
			size_t N = input_ptr->num_rows();

			for(size_t out = 0; out < num_out; ++out){
				for(size_t i = 0; i < N; ++i){
					for(size_t j = 0; j < N; ++j){
						//	dy_j/dx_i = [(j == i - 1/N)(std) - (x_j - mean)(1/(N * std) * (x_i - mean))]/[std^2]
						double dy_j_dx_i = 
						( ((i == j) - 1./N) * __std[0][out] - ((*input_ptr)[j][out] - __mean[0][out]) 
							* ((*input_ptr)[i][out] - __mean[0][out])/(N * __std[0][out]) )
						/ 
						(std::pow(__std[0][out], 2.0));

						grad_push[i][out] += grad_current[j][out] * dy_j_dx_i;
					}
				}
			}

			input_ptr->backward(grad_push);
		}

		void zero_grad(){
			input_ptr->zero_grad();
		}

		void make_step(double step){
			input_ptr->make_step(step);
		}

		void break_graph(){
			input_ptr->break_graph();
		}


		~BatchNorm() = default;
	};


	template<size_t In, size_t Out>
	class Linear: public nnLayer{
	/*
		x: N x In
		w: In x Out
		b: 1 x Out
		y: N x Out

		
		let after it will be f(y) and we have grad_y where y = x*w + b
		so we need to find dy/dw
	
		from Adder:
			grad_b = grad_y
			grad_{xw} = grad_y

		from Multiplier:
			grad_x = 1/Out * grad_y * w^T
			grad_w = 1/N * x^T * grad_y
	*/

	private:
		Matrix<double>* input_ptr;

		Matrix<double> w;
		Matrix<double> b;

	public:
		//TODO
		Linear(): w(Matrix<double>::random(In, Out, nullptr)), b(Matrix<double>::random(1, Out, nullptr)){}
		// Linear(): w(Matrix<double>(In, Out, 1.0, nullptr)), b(Matrix<double>(1, Out, 1.0, nullptr)){}


		Linear(const Linear& other) = delete;
		Linear& operator=(const Linear& other) = delete;

		Linear(Linear&& other) = default;
		Linear& operator=(Linear&& other) = default;

		Matrix<double> forward(Matrix<double>& input) override{
			input_ptr = &input;

			Matrix<double> result = input;
			result.layer_ptr.reset();
			result *= w;

			//	+= b
			for(size_t num = 0; num < input.num_rows(); ++num){
				for(size_t i = 0; i < Out; ++i){
					result[num][i] += b[0][i];
				}
			}

			result.layer_ptr = shared_from_this();
			return result;
		}
		
		void backward() override{
			// grad_A = grad_C * B^T
			// grad_B = A^T * grad_C
			// x * w + b

			size_t N = input_ptr->num_rows();

			auto& grad_current = *(res_ptr->grad_ptr);

			Matrix<double> grad_push = grad_current; grad_push *= w.transpose();
			Matrix<double> grad_w = input_ptr->transpose(); grad_w *= grad_current;
			Matrix<double> grad_b = Matrix<double>(1, N, 1.0); grad_b *= grad_current;

			w.layer_ptr.reset();
			b.layer_ptr.reset();

			w.backward(grad_w);
			b.backward(grad_b);

			input_ptr->backward(grad_push);
		}

		void zero_grad(){
			w.zero_grad();
			b.zero_grad();
			input_ptr->zero_grad();
		}

		void make_step(double step){
			auto& grad_b = *(b.grad_ptr); 	auto& grad_w = (*w.grad_ptr);
			grad_b *= step; 				grad_w *= step;
			b -= grad_b; 					w -= grad_w;

			input_ptr->make_step(step);
		}

		void break_graph(){
			input_ptr->break_graph();
		}

		Matrix<double> get_weight() const{
			return w;
		}

		Matrix<double> get_bias() const{
			return b;
		}

		~Linear() = default;
	};

	
	template<typename... Layers>
	class Sequential{
	private:
		std::list<std::shared_ptr<nnLayer>> seq;
		std::list<Matrix<double>> outputs;

		template<typename... LLayers>
		void Add(LLayers&&... layers){}

		template<typename Head, typename... LLayers>
		void Add(){
			seq.push_back(std::make_shared<Head>());
			Add<LLayers...>();
		}

	public:
		static const size_t length = sizeof...(Layers);

		Sequential(){
			Add<Layers...>();
		}

		Matrix<double> forward(const Matrix<double>& input){
			outputs.clear();
			outputs.push_back(input);
			for(auto& layer_ptr: seq){
				auto& input_current = outputs.back();
				auto&& output_current = layer_ptr->forward(input_current);
				outputs.push_back(std::move(output_current));
			}
			return outputs.back();
		}

		void backward(const Matrix<double>& grad_other){
			outputs.back().backward(grad_other);
		}

		void make_step(double step){
			outputs.back().make_step(step);
		}

		void zero_grad(){
			outputs.back().zero_grad();
		}

		void break_graph(){
			outputs.back().break_graph();
		}

		~Sequential(){}
	};


	//Losses
	class MSELoss: public nnLayer{
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


	public:
		MSELoss() = default;
		MSELoss(const MSELoss& other) = delete;
		MSELoss& operator=(const MSELoss& other) = delete;
		MSELoss(MSELoss&& other) = default;
		MSELoss& operator=(MSELoss&& other) = default;

		Matrix<double> forward(Matrix<double>& real, Matrix<double>& pred) override{
			if(real.size() != pred.size()){
				throw BadShape("Wrong sizes. MSELoss, forward");
			}

			pred_ptr = &pred;
			real_ptr = &real;

			Matrix<double> result(1, 1, 0, shared_from_this());

			for(size_t i = 0; i < real.num_rows(); ++i){
				for(size_t j = 0; j < real.num_columns(); ++j){
					result[0][0] += std::pow(real[i][j] - pred[i][j], 2);
				}
			}
			return result;
		}

		void backward() override{
			Matrix<double> grad_push(*pred_ptr);
			grad_push -= *real_ptr;
			grad_push *= 2;
			grad_push /= pred_ptr->num_rows();

			pred_ptr->backward(grad_push);
		}

		void make_step(double step){
			pred_ptr->make_step(step);
		}

		void zero_grad(){
			pred_ptr->zero_grad();
		}

		void break_graph(){
			pred_ptr->break_graph();
			real_ptr->break_graph();
		}

		~MSELoss() = default;
	};

	template<size_t Classes>
	Matrix<bool> OneHot(const Matrix<double>& input){
		Matrix<bool> result(input.num_rows(), Classes, 0);
		for(size_t i = 0; i < input.num_rows(); ++i){
			result[i][input[i][0]] = 1;
		}
		return result;
	}

	template<size_t Classes>
	Matrix<size_t> predict(const Matrix<double>& input){
		Matrix<size_t> result(input.num_rows(), 1, 0);
		for(size_t num = 0; num < input.num_rows(); ++num){
			size_t index_max = 0;
			for(size_t i = 0; i < Classes; ++i){
				if(input[num][index_max] < input[num][i]){
					index_max = i;
				}
			}
			result[num][0] = index_max;
		}
		return result;
	}

	Matrix<double> accuracy(const Matrix<double>& real, const Matrix<double>& predicted_classes){
		Matrix<double> result(1, 1, 0.0);
		result[0][0] = (real == predicted_classes).sum();
		result[0][0] /= real.num_rows();
		return result;
	}


	template<size_t Classes>
	class CrossEntropyLoss: public nnLayer{
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
		Matrix<double>* real_ptr;
		Matrix<double>* pred_ptr;
		Matrix<double> logits;

	public:

		CrossEntropyLoss() = default;

		Matrix<double> forward(Matrix<double>& real, Matrix<double>& pred) override{
			if(real.size() != pred.size()){
				throw BadShape("Wrong sizes of matrices. CrossEntropyLoss, forward");
			}

			if(real.num_columns() != Classes){
				throw BadShape("Wrong size of matrix, should be N x Classes. CrossEntropyLoss, forward");
			}

			real_ptr = &real;
			pred_ptr = &pred;

			logits = Matrix<double>(pred.size(), 0.0);
			for(size_t num = 0; num < pred.num_rows(); ++num){
				double _sum = 0.0;
				for(size_t i = 0; i < Classes; ++i){
					_sum += std::exp(pred[num][i]);
				}

				for(size_t i = 0; i < Classes; ++i){
					logits[num][i] = std::exp(pred[num][i]) / _sum;
				}
			}

			Matrix<double> result(1, 1, 0.0, shared_from_this());

			for(size_t num = 0; num < pred.num_rows(); ++num){
				for(size_t i = 0; i < Classes; ++i){
					result[0][0] += std::log(logits[num][i]) * real[num][i];
				}
				result[0][0] *= -1;
			}
			return result;
		}

		void backward(){
			Matrix<double> grad_push(logits);
			grad_push -= *real_ptr;
			grad_push /= pred_ptr->num_rows();
			pred_ptr->backward(grad_push);
		}

		void make_step(double step){
			pred_ptr->make_step(step);
		}

		void zero_grad(){
			pred_ptr->zero_grad();
		}

		void break_graph(){
			pred_ptr->break_graph();
		}
	};
}

