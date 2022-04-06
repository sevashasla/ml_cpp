#pragma once

#include "../matrix/matrix.hpp"

namespace ml::linear_models{

template<size_t M>
class LinearRegression{
/*
	y: 1 x n
	x: m x n
	W: 1 x m

	f = (y - Wx) * (y - Wx)^T
	g = y - Wx
	p = (y - Wx)^T

	df/dW(W_0) = dg/dW(W_0) * p + g * dp/dW(W_0)
	dg/dW(W_0) = dg_{W_0}(W)
	g(a + h) - g(a) = dg_a(h) + ... => dg_a(h) = hb
						or
	[R^{1 x n}] = h * D_g => D_g: R^{m x n} and
	D_g = 
	(dg1/dw1, ..., dgn/dw1)
		...
	(dg1/dwm, ..., dgn/dwm)
	(this is not so obvious)
	and one can get below analogously
	dp_a(h) = (hb)^T
	therefore
	
	df/dW(W_0) = -W_0 x * (y - W_0 x)^T + (y - W_0 x) * (-W_0 x)^T = 
	= -W_0 x * y^T + W_0 x * (W_0 x)^T - y * (W_0 x)^T + W_0 x * (W_0 x)^T = 
	= 2 W_0 * x * x^T * W_0^T - y * (W_0 x)^T - (W_0 x) * y^T = (size is 1 x 1) = 
	= 2 W_0 * x * x^T * W_0^T - 2 (W_0 x) * y^T = 2 (W_0 x) * (W_0 x - y)^T
	=>
	grad = 2 * (W_0 x - y) * x^T
*/

private:
	Matrix<double> w;

	auto& add_ones(Matrix<double>& data) const{
		size_t N = data.num_columns();

		if(data.num_rows() != M){
			throw BadShape("wrong shape in add_ones");
		}

		data.push_back(std::vector<double>(N, 1.0));
		return data;
	}

	auto add_ones(const Matrix<double>& data) const{
		size_t N = data.num_columns();

		if(data.num_rows() != M){
			throw BadShape("wrong shape in add_ones");
		}


		Matrix res(M + 1, N, 0.0);
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < N; ++j){
				res[i][j] = data[i][j];
			}
		}
		for(size_t j = 0; j < N; ++j){
			res[M][j] = 1.0;
		}
		return res;
	}

public:
	LinearRegression() : w(Matrix<double>::random(1, M + 1)){}


	void train(const Matrix<double>& y, const Matrix<double>& x_fresh, size_t num_epoch=1000, double step=1e-3){
		size_t N = y.num_columns();

		if(x_fresh.num_rows() != M){
			throw BadShape("wrong size x_fresh in train");
		}

		auto x = add_ones(x_fresh);
		while(num_epoch){
			if(num_epoch % 100 == 0){
				std::cout << get_error(y, x_fresh) << "\n";
			}
			--num_epoch;

			auto grad = (w * x - y) * x.transpose();
			grad *= step;
			w -= grad;
		}
	}


	auto predict(const Matrix<double>& data) const{
		if(data.num_rows() != M){
			throw BadShape("wrong size of data in predict");
		}
		return w * add_ones(data);
	}

	
	auto get_error(const Matrix<double>& y, const Matrix<double>& x_fresh) const{
		auto tmp = (y - predict(x_fresh));
		return (tmp * tmp.transpose())[0][0];
	}

	auto get_weight() const{
		return w;
	}
};



template<size_t Classes, size_t M>
class LogisticRegression{
/*	
	
	but I don't know why it's not good to use another loss-f like:
		cross_entropy' = -sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))

	y: C x N
	x: M x N
	w: C x M

	loss = -(sum(q_i * log(p_i) + (1 - q_i) * log(1 - p_i)))
	z_i - out of the layer num. i

	dlog(p_i) / dz_j = d/dz_j(log(e^{z_i} / sum(e^{z_k}))) = d/dz_j(z_i) - d/dz_j(log(sum(e^{z_k}))) = 
	= (j == i ? 1 : 0) - 1/sum(e^{z_k}) * e^{z_j} = (j == i ? 1 : 0) - p_j = 
	= (j == i ? 1 - p_j : -p_j)
	
	let's look closely at only one vector
	
	(z_1)			(p_1)			log(p_1)
	...		->	P = ...		-> y^T * 	...	=(y_1 * log(p_1) + ... + y_C * log(p_C))
	(z_C)			(p_C)			log(p_C)

	and df/dz_1 = y_1 * (1 - p_1) + y_2 * -p_1 + ... + y_C * -p_1 = (y_1 == 1 ? 1 - p_1 : -p_1) = y_1 - p_1

	(w_{1 1}, ... , w_{1 M})		(x_1)		(w_{1 1} * x_1 + ... + w_{1 M} * x_M)	= z_1
				...				* 	...		= 	...										
	(w_{C 1}, ... , w_{C M})		(x_M)		(w_{C 1} * x_1 + ... + w_{C M} * x_M)	= z_C

	df/dw_{1 1} = df/dz_1 * dz_1/dw_{1 1} = df/dz_1 * x_1, ..., df/dw_{1 M} = df/dz_1 * x_M
	=>
	matrix of grad is equal to:
	(df/dz_1 * x_1, ... , df/dz_1 * x_M)	(df/dz_1)
					...					=		...		* (x_1, ... , x_M)
	(df/dz_C * x_1, ... , df/dz_c * x_M)	(df/dz_c)	

	so it is grad_z * x^T
*/

private:
	Matrix<double> w;

	
	auto softmax(const Matrix<double>& z) const{
		size_t N = z.num_columns();
		if(z.num_rows() != Classes){
			throw BadShape("wrong size of z in softmax");
		}

		Matrix<double> res(Classes, N, 0.0);
		for(size_t i = 0; i < N; ++i){
			double sum = 0.0;
			for(size_t j = 0; j < Classes; ++j){
				sum += std::exp(z[j][i]);
			}
			for(size_t j = 0; j < Classes; ++j){
				res[j][i] = std::exp(z[j][i]) / sum;
			}
		}
		return res;
	}

	auto cross_entropy(const Matrix<double>& y, const Matrix<double>& outs){
		if(y.size() != outs.size()){
			throw BadShape("y and out must have equal size");
		}

		size_t N = y.num_columns();

		double res = 0.0;
		for(size_t i = 0; i < N; ++i){
			for(size_t j = 0; j < Classes; ++j){
				res -= y[j][i] * std::log(outs[j][i]);
			}
		}
		return res;
	}

	
	auto add_ones(const Matrix<double>& data) const{
		if(data.num_rows() != M){
			throw BadShape("wrong size of data in add_ones");
		}

		size_t N = data.num_columns();
		Matrix<double> res(M + 1, N, 0.0);
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < N; ++j){
				res[i][j] = data[i][j];
			}
		}
		for(size_t j = 0; j < N; ++j){
			res[M][j] = 1.0;
		}
		return res;
	}

	auto indicate(const Matrix<double>& data) const{
		size_t N = data.num_columns();
		Matrix<double> res(Classes, N, 0.0);
		for(size_t i = 0; i < N; ++i){
			res[data[0][i]][i] = 1;
		}
		return res;
	}

	
	auto get_grad(const Matrix<double>& y, const Matrix<double>& x) const{
		if(y.num_rows() != Classes){
			throw BadShape("wrong size of y in get_grad");
		}

		if(x.num_rows() != M + 1){
			throw BadShape("wrong size of x in get_grad");
		}

		size_t N = y.num_columns();
		Matrix<double> grad_z(Classes, N, 0.0);
		auto p = softmax(w * x);
		grad_z = p - y;
		auto grad = grad_z * x.transpose();
		grad /= N; //mean
		return grad;
	}


public:
	LogisticRegression(): w(Matrix<double>::random(Classes, M + 1)){}

	void train(const Matrix<double>& y_fresh, const Matrix<double>& x_fresh, size_t num_epoch=1000, double step=1e-3){
		size_t N = y_fresh.num_columns();

		if(x_fresh.num_rows() != M){
			throw BadShape("wrong size of x_fresh in train");
		}

		auto x = add_ones(x_fresh);
		auto y = indicate(y_fresh);
		while(num_epoch){
			if(num_epoch % 100 == 0){
				std::cout << get_error(y_fresh, x_fresh) << "\n";
			}	
			--num_epoch;

			auto grad = get_grad(y, x);
			grad *= step;
			w -= grad;
		}
	}

	
	auto predict(const Matrix<double>& x_fresh){
		size_t N = x_fresh.num_columns();
		
		if(x_fresh.num_rows() != M){
			throw BadShape("wrong shape of x in predict");
		}

		auto x = add_ones(x_fresh);
		auto out = w * x;
		Matrix<double> res(1, N, 0.0);
		for(size_t num = 0; num < N; ++num){
			size_t index_max = 0;
			for(size_t j = 0; j < Classes; ++j){
				if(out[j][num] > out[index_max][num]){
					index_max = j;
				}
			}
			res[0][num] = index_max;
		}
		return res;
	}

	
	auto get_error(const Matrix<double>& y_fresh, const Matrix<double>& x_fresh){

		if(x_fresh.num_rows() != M){
			throw BadShape("wrong shape of x_fresh in get_error");
		}

		size_t N = x_fresh.num_columns();

		auto x = add_ones(x_fresh);
		auto y = indicate(y_fresh);
		return cross_entropy(y, softmax(w * x));
	}


	auto get_accuracy(const Matrix<double>& y_fresh, const Matrix<double>& x_fresh){
		if(x_fresh.num_rows() != M){
			throw BadShape("wrong shape of x_fresh in get_accuracy");
		}

		size_t N = x_fresh.num_columns();

		auto x = add_ones(x_fresh);
		auto y = indicate(y_fresh);
		return (y_fresh == predict(x_fresh)).sum() / N;
	}

	auto get_weight() const{
		return w;
	}
};

} // end of ml::linear_models
