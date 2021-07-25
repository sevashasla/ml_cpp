#pragma once

#include "matrix.h"
#include <list>
#include <memory>


template<typename...>
void f() = delete;
/*
class Layer: public std::enable_shared_from_this<Layer>{
public:
	Matrix<double>* res_ptr=nullptr;
	virtual void backward(const Matrix<double>&) = 0;
	virtual Matrix<double> forward(Matrix<double>&) = 0;
	virtual void make_step(double) = 0;
	virtual void zero_grad() = 0;
	virtual void change_res_ptr(Matrix<double>*) = 0;
	virtual void break_graph() = 0;
	virtual ~Layer() = default;
};
*/

namespace nn{
	using nnLayer = Matrix<double>::Layer;
	

	class ReLU: public nnLayer{
	private:
		Matrix<double>* res_ptr=nullptr;
		Matrix<double>* input_ptr=nullptr;
		Matrix<bool> mask;

		virtual void change_res_ptr(Matrix<double>* ptr) {
			res_ptr = ptr;
		}

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

		void backward(const Matrix<double>& grad_other) override{
			auto& grad_current = *(res_ptr->grad_ptr);
			
			Matrix<double> grad_push(grad_other);
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
		Matrix<double>* res_ptr=nullptr;

		void change_res_ptr(Matrix<double>* ptr){
			res_ptr = ptr;
		}

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

			Matrix<double> result(1, 1, 0, nullptr);
			result.layer_ptr = shared_from_this();
			

			for(size_t i = 0; i < real.num_rows(); ++i){
				for(size_t j = 0; j < real.num_columns(); ++j){
					result[0][0] += std::pow(real[i][j] - pred[i][j], 2);
				}
			}
			return result;
		}

		void backward(const Matrix<double>& _not_used=Matrix<double>()) override{
			//I don't need to change_crad current

			Matrix<double> grad_push(*pred_ptr);
			grad_push -= *real_ptr;
			grad_push *= 2;

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
		Matrix<double>* res_ptr;
		Matrix<double>* input_ptr;

		Matrix<double> w;
		Matrix<double> b;

		void change_res_ptr(Matrix<double>* ptr){
			res_ptr = ptr;
		}

	public:
		Linear(): w(Matrix<double>::random(In, Out, nullptr)), b(Matrix<double>::random(1, Out, nullptr)){}
		Linear(const Linear& other) = delete;
		Linear& operator=(const Linear& other) = delete;

		Linear(Linear&& other) = default;
		Linear& operator=(Linear&& other) = default;

		Matrix<double> forward(Matrix<double>& input) override{
			input_ptr = &input;

			Matrix<double> result = input;
			result.layer_ptr.reset();
			result *= w;
			for(size_t num = 0; num < input.num_columns(); ++num){
				for(size_t i = 0; i < Out; ++i){
					result[num][i] += b[0][i];
				}
			}

			result.layer_ptr = shared_from_this();
			return result;
		}
		
		void backward(const Matrix<double>& grad_other) override{
			// grad_A = 1/K * grad_f * B^T
			// grad_B = 1/M * A^T * grad_f

			size_t N = input_ptr->num_rows();

			Matrix<double> grad_push = grad_other; grad_push *= w.transpose(); grad_push /= Out;
			Matrix<double> grad_w = input_ptr->transpose(); grad_w *= grad_other; grad_w /= N;
			Matrix<double> grad_b = Matrix<double>(1, N, 1.0, nullptr); grad_b *= grad_other; grad_b /= N;

			b.layer_ptr.reset();
			w.layer_ptr.reset();

			b.backward(grad_b);
			w.backward(grad_w);

			input_ptr->backward(grad_push);
		}

		void zero_grad(){
			w.zero_grad();
			b.zero_grad();
			input_ptr->zero_grad();
		}

		void make_step(double step){
			auto grad_b = b.get_gradient(); 	auto grad_w = w.get_gradient();
			grad_b *= step; 					grad_w *= step;
			b -= grad_b; 						w -= grad_w;

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

		void backward(Matrix<double>& grad_other){
			seq.back()->backward(grad_other);
		}

		void make_step(double step){
			seq.back()->make_step(step);
		}

		void zero_grad(){
			seq.back()->zero_grad();
		}

		void break_graph(){
			seq.back()->break_graph();
		}

		~Sequential(){}
	};

}

