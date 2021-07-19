#include "matrix_template.h"

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
	Matrix<1, M + 1> w;

	template<size_t N>
	auto add_ones(const Matrix<M, N>& data) const{
		Matrix<M + 1, N> res;
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
	LinearRegression() : w(Matrix<1, M + 1>::random()){}

	template<size_t N>
	void train(const Matrix<1, N>& y, const Matrix<M, N>& x_fresh, size_t num_epoch=1000, double step=1e-3){
		auto x = add_ones(x_fresh);
		while(num_epoch){
			--num_epoch;
			auto grad = (w * x - y) * x.transpose();
			grad *= step;
			w -= grad;
		}
	}

	template<size_t N>
	auto predict(const Matrix<M, N>& data) const{
		return w * add_ones(data);
	}

	template<size_t N>
	auto get_error(const Matrix<1, N>& other_y, const Matrix<M, N>& other_x) const{
		auto tmp = (other_y - w * add_ones(other_x));
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
	Matrix<Classes, M + 1> w;

	template<size_t N>
	auto softmax(const Matrix<Classes, N>& z) const{
		Matrix<Classes, N> res(0.0);
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

	template<size_t N>
	auto cross_entropy(const Matrix<Classes, N>& y, const Matrix<Classes, N>& outs){
		double res = 0.0;
		for(size_t i = 0; i < N; ++i){
			for(size_t j = 0; j < Classes; ++j){
				res -= y[j][i] * std::log(outs[j][i]);
			}
		}
		return res;
	}

	template<size_t N>
	auto add_ones(const Matrix<M, N>& data) const{
		Matrix<M + 1, N> res;
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

	template<size_t N>
	auto indicate(const Matrix<1, N>& data) const{
		Matrix<Classes, N> res(0.0);
		for(size_t i = 0; i < N; ++i){
			res[data[0][i]][i] = 1;
		}
		return res;
	}

	template<size_t N>
	auto get_grad(const Matrix<Classes, N>& y, const Matrix<M + 1, N>& x) const{
		Matrix<Classes, N> grad_z(0.0);
		Matrix<Classes, N> p = softmax(w * x);
		grad_z = p - y;
		auto grad = grad_z * x.transpose();
		grad /= N; //mean
		return grad;
	}


public:
	LogisticRegression(): w(Matrix<Classes, M + 1>::random()){}

	template<size_t N>
	void train(const Matrix<1, N>& y_fresh, const Matrix<M, N>& x_fresh, size_t num_epoch=1000, double step=1e-3){
		auto x = add_ones(x_fresh);
		auto y = indicate(y_fresh);
		while(num_epoch){
			if(num_epoch % 100 == 0){
				cout << get_error(y_fresh, x_fresh) << "\n";
			}	
			--num_epoch;

			auto grad = get_grad(y, x);
			grad *= step;
			w -= grad;
		}
	}

	template<size_t N>
	auto predict(const Matrix<M, N>& x_fresh){
		auto x = add_ones(x_fresh);
		auto out = w * x;
		Matrix<1, N> res;
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

	template<size_t N>
	auto get_error(const Matrix<1, N>& y_fresh, const Matrix<M, N>& x_fresh){
		auto x = add_ones(x_fresh);
		auto y = indicate(y_fresh);
		return cross_entropy(y, softmax(w * x));
	}


	template<size_t N>
	auto get_accuracy(const Matrix<1, N>& y_fresh, const Matrix<M, N>& x_fresh){
		auto x = add_ones(x_fresh);
		auto y = indicate(y_fresh);
		return (y_fresh == predict(x_fresh)).sum() / N;
	}

	auto get_weight() const{
		return w;
	}
};
