#pragma once 

#include "../tensor/tensor.hpp"
#include "../tensor/basic_layer.hpp"

namespace ml::nn::layers {

template<typename Field>
class ReLU: public Layer<Field>{
private:
	Tensor<Field>* input_ptr=nullptr;
	Matrix<bool> mask;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:
	ReLU() = default;

	Tensor<Field> forward(Tensor<Field>& input) override{
		input_ptr = &input;
		mask = Matrix<bool>(input.size(), 0);
		
		Tensor<Field> result = Tensor<Field>(
			static_cast<Matrix<Field>&>(input), 
			SharedBase::shared_from_this()
		);

		for(size_t i = 0; i < result.num_rows(); ++i){
			for(size_t j = 0; j < result.num_columns(); ++j){
				mask[i][j] = (result[i][j] > 0.0);
				result[i][j] = std::max(result[i][j], 0.0);
			}
		}
		return result;
	}

	void backward_(const Matrix<Field>& grad) override{
		Matrix<Field> grad_push(grad);
		for(size_t i = 0; i < grad_push.num_rows(); ++i){
			for(size_t j = 0; j < grad_push.num_columns(); ++j){
				grad_push[i][j] *= mask[i][j];
			}
		}
		input_ptr->backward(grad_push);
	}

	void make_step_(Field step){
		input_ptr->make_step(step);
	}

	void zero_grad_(){
		input_ptr->zero_grad();
	}

	void break_graph_() override{
		input_ptr->break_graph();
	}
};

template<typename Field>
class LeakyReLU: public Layer<Field>{
private:
	Tensor<Field>* input_ptr=nullptr;
	Matrix<Field> mask;
	Field alpha=0;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:
	LeakyReLU(Field alpha=1e-2): alpha(alpha){}

	Tensor<Field> forward(Tensor<Field>& input) override{
		input_ptr = &input;
		mask = Matrix<Field>(input.size(), 0);
		
		Tensor<Field> result = Tensor<Field>(
			static_cast<Matrix<Field>&>(input), 
			SharedBase::shared_from_this()
		);

		for(size_t i = 0; i < result.num_rows(); ++i){
			for(size_t j = 0; j < result.num_columns(); ++j){
				mask[i][j] = (result[i][j] > 0.0 ? 1 : alpha);
				result[i][j] = std::max(result[i][j], alpha * result[i][j]);
			}
		}
		return result;
	}

	void backward_(const Matrix<Field>& grad) override{
		Matrix<double> grad_push(grad);
		for(size_t i = 0; i < grad_push.num_rows(); ++i){
			for(size_t j = 0; j < grad_push.num_columns(); ++j){
				grad_push[i][j] *= mask[i][j];
			}
		}
		input_ptr->backward(grad_push);
	}

	void make_step_(Field step){
		input_ptr->make_step(step);
	}

	void zero_grad_(){
		input_ptr->zero_grad();
	}

	void break_graph_() override{
		input_ptr->break_graph();
	}
};


template<typename Field>
class Sigmoid: public Layer<Field>{
/*
	y = 1/(1 + e^-x)
	df/dx = d(1/(1 + e^-x))/dx = -1/(1 + e^-x)^2 * -e^-x = 
	= e^-x/(1 + e^-x)^2 = y(1 - y)
	I have df/dy => df/dx = df/dy * dy/dx
*/

private:
	Tensor<Field>* input_ptr;
	Matrix<Field> result;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:

	Tensor<Field> forward(Tensor<Field>& input){
		input_ptr = &input;

		Tensor<Field> res(input.size(), SharedBase::shared_from_this());
		for(size_t num = 0; num < input.num_rows(); ++num){
			for(size_t i = 0; i < input.num_columns(); ++i){
				res[num][i] = 1.0 / (1.0 + std::exp(-input[num][i]));
			}
		}
		result = static_cast<Matrix<Field>&>(result);
		return res;
	}


	void backward_(const Matrix<Field>& grad) override {
		Matrix<Field> grad_push(grad.size(), 0.0);
		for(size_t num = 0; num < grad.num_rows(); ++num){
			for(size_t i = 0; i < grad.num_columns(); ++i){
				grad_push[num][i] = grad[num][i] * result[num][i] * (1 - result[num][i]);
			}
		}
		input_ptr->backward(grad_push);
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

template<typename Field>
class BatchNorm: public Layer<Field> {
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
	Tensor<Field>* input_ptr;
	Matrix<Field> mean_;
	Matrix<Field> std_;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:
	BatchNorm() = default;

	static Matrix<Field> mean(const Matrix<Field>& input){
		size_t num_out = input.num_columns();
		size_t N = input.num_rows();

		Matrix<Field> result(1, num_out, 0.0);
		for(size_t num = 0; num < N; ++num){
			for(size_t i = 0; i < num_out; ++i){
				result[0][i] += input[num][i];
			}
		}
		result /= N;
		return result;
	}

	static Matrix<Field> std(const Matrix<Field>& input){
		size_t num_out = input.num_columns();
		size_t N = input.num_rows();

		Matrix<Field> result(1, num_out, 0.0);
		Matrix<Field> mean_ = mean(input);

		for(size_t num = 0; num < N; ++num){
			for(size_t i = 0; i < num_out; ++i){
				result[0][i] += std::pow(input[num][i] - mean_[0][i], 2);
			}
		}
		result /= N;
		for(size_t i = 0; i < num_out; ++i){
			result[0][i] = std::sqrt(result[0][i]);
		}
		return result;
	}

	Tensor<Field> forward(Tensor<Field>& input){
		input_ptr = &input;

		mean_ = mean(input);
		std_ = std(input);

		Matrix<Field> result(input);

		for(size_t num = 0; num < input.num_rows(); ++num){
			for(size_t i = 0; i < input.num_columns(); ++i){
				result[num][i] -= mean_[0][i];
				result[num][i] /= std_[0][i];
			}
		}

		Tensor<Field> result_tensor(result, SharedBase::shared_from_this());

		return result_tensor;
	}

	void backward_(const Matrix<Field>& grad){
		Matrix<Field> grad_push(grad.size(), 0.0);
		size_t num_out = input_ptr->num_columns();
		size_t N = input_ptr->num_rows();

		for(size_t out = 0; out < num_out; ++out){
			for(size_t i = 0; i < N; ++i){
				for(size_t j = 0; j < N; ++j){
					//	dy_j/dx_i = [(j == i - 1/N)(std) - (x_j - mean)(1/(N * std) * (x_i - mean))]/[std^2]
					Field dy_j_dx_i = 
					( ((i == j) - 1./N) * std_[0][out] - ((*input_ptr)[j][out] - mean_[0][out]) 
						* ((*input_ptr)[i][out] - mean_[0][out])/(N * std_[0][out]) )
					/ 
					(std::pow(std_[0][out], 2.0));

					grad_push[i][out] += grad[j][out] * dy_j_dx_i;
				}
			}
		}

		input_ptr->backward(grad_push);
	}

	void zero_grad_(){
		input_ptr->zero_grad();
	}

	void make_step_(Field step){
		input_ptr->make_step(step);
	}

	void break_graph_(){
		input_ptr->break_graph();
	}


	~BatchNorm() = default;
};


template<typename Field, size_t In, size_t Out>
class Linear: public Layer<Field> {
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
	Tensor<Field>* input_ptr;
	Tensor<Field> w;
	Tensor<Field> b;
	using SharedBase = std::enable_shared_from_this<Layer<Field>>;

public:
	//TODO
	Linear(): w(Matrix<Field>::random(In, Out), nullptr), b(Matrix<Field>::random(1, Out), nullptr){}
	// Linear(): w(Matrix<Field>(In, Out, 0.4)), b(Matrix<Field>(1, Out, 0.2), nullptr){}

	Tensor<Field> forward(Tensor<Field>& input) override{
		input_ptr = &input;

		Matrix<Field> result = input;
		result *= w;

		//	+= b
		for(size_t num = 0; num < input.num_rows(); ++num){
			for(size_t i = 0; i < Out; ++i){
				result[num][i] += b[0][i];
			}
		}

		Tensor<Field> result_tensor(result, SharedBase::shared_from_this());
		return result_tensor;
	}
	
	void backward_(const Matrix<Field>& grad) override {
		// grad_A = grad_C * B^T
		// grad_B = A^T * grad_C
		// x * w + b

		size_t N = input_ptr->num_rows();

		Matrix<Field> grad_push = grad; grad_push *= w.transpose();
		Matrix<Field> grad_w = input_ptr->transpose(); grad_w *= grad;
		Matrix<Field> grad_b = Matrix<Field>(1, N, 1.0); grad_b *= grad;

		w.backward(grad_w);
		b.backward(grad_b);

		input_ptr->backward(grad_push);
	}

	void zero_grad_() override {
		w.zero_grad();
		b.zero_grad();
		input_ptr->zero_grad();
	}

	void make_step_(Field step) override {
		w.make_step(step);
		b.make_step(step);
		input_ptr->make_step(step);
	}

	void break_graph_() override {
		input_ptr->break_graph();
	}

	Matrix<Field> get_weight() const{
		return w;
	}

	Matrix<Field> get_bias() const{
		return b;
	}

	~Linear() = default;
};
} // end of namespace ml::nn::layers