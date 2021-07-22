#pragma once

#include <iostream>
#include <type_traits>
#include <algorithm>
#include <cmath>


template<size_t M, size_t N, typename Field=double>
class Matrix{
	//is it good to add allocator?

	template<size_t MM, size_t NN, typename FField>
	friend std::ostream& operator<<(std::ostream& out, const Matrix<MM, NN, FField>& m);

	template<size_t MM, size_t NN, typename FField>
	friend std::istream& operator>>(std::istream& out, Matrix<MM, NN, FField>& m);


private:
	Field epsilon = 1e-3;
	std::vector<std::vector<Field>> m;

	// second = first * coeff + second
	void add_str(size_t first, size_t second, Field coeff = 1){
		for(size_t k = 0; k < N; ++k){
			m[second][k] += coeff * m[first][k];
		}
	}

	void add_row(size_t first, size_t second, Field coeff = 1){
		for(size_t k = 0; k < M; ++k){
			m[k][second] += coeff * m[k][first];
		}
	}

	bool equal(const Field& left, const Field& right) const{
		return (-epsilon <= right - left) && (right - left <= epsilon);
	}


public:
	Matrix(): m(M, std::vector<Field>(N, 0)){
		for(size_t i = 0; i < std::min(M, N); ++i){
			m[i][i] = 1;
		}
	}

	static Matrix<M, N, Field> eye(){
		return Matrix<M, N>();
	}

	static Matrix<M, N, Field> random(){
		Matrix<M, N> res(0);
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < N; ++j){
				res[i][j] = static_cast<Field>(rand()) / static_cast<Field>(RAND_MAX);
			}
		}
		return res;
	}


	Matrix(const Field& f): m(M, std::vector<Field>(N, f)){}
	Matrix(const Matrix&) = default;
	Matrix(Matrix&&) = default;
	
	Matrix(const std::vector<std::vector<Field>>& m_other): m(m_other){}
	Matrix(std::vector<std::vector<Field>>&& m_other): m(std::move(m_other)){}

	Matrix& operator=(const Matrix& other) & {
		m = other.m;
		return *this;
	}

	Matrix& operator=(Matrix&& other) & {
		m = std::move(other.m);
		return *this;
	}

	Field norm(double p, double q) const{//I haven't check it yet
		Field res = 0;
		for(auto& row: m){
			Field row_res = 0;
			for(auto& elem: row){
				row_res += std::pow(elem, p);
			}
			res += std::pow(row_res, q / p);
		}
		res = std::pow(res, 1 / q);
	};

	std::vector<Field>& operator[](size_t i){
		return m[i];
	}

	const std::vector<Field>& operator[](size_t i) const {
		return m[i];
	}

	Matrix& operator*=(const Field& f){
		for(auto& row: m){
			for(auto& elem: row){
				elem *= f;
			}
		}
		return *this;
	}

	Matrix& operator/=(const Field& f){
		for(auto& row: m){
			for(auto& elem: row){
				elem /= f;
			}
		}
		return *this;
	}

	Matrix& operator*=(const Matrix<M, N, Field>& other){
		static_assert(N == M, "only for square matrix");
		Matrix<N, M, Field> result(0);
		for(size_t k = 0; k < M; ++k){
			for(size_t i = 0; i < N; ++i){
				for(size_t j = 0; j < M; ++j){
					result[i][j] += m[i][k] * other[k][j];
				}
			}
		}

		*this = std::move(result);
		return *this;
	}

	Matrix& operator+=(const Matrix<M, N, Field>& other){
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < N; ++j){
				m[i][j] += other[i][j];
			}
		}
		return *this;
	}

	Matrix& operator-=(const Matrix<M, N, Field>& other){
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < N; ++j){
				m[i][j] -= other[i][j];
			}
		}
		return *this;
	}

	Field& tr() const{
		Field res = 0;
		for(size_t i = 0; i < std::min(N, M); ++i){
			res += m[i][i];
		}
		return res;
	}

	Matrix<N, M, Field> transpose() const{
		Matrix<N, M, Field> transposed(0);
		for(size_t i = 0; i < N; ++i){
			for(size_t j = 0; j < M; ++j){
				transposed[i][j] = m[j][i];
			}
		}
		return transposed;
	}


	Field det() const{
		static_assert(M == N, "Only square matrix");
		Field res = 1.0;
		Matrix<M, N, Field> _copy(m);
		bool found = true;

		for(size_t i = 0; i < M; ++i){

			if(equal(_copy[i][i], 0.0)){
				found = false;
				for(size_t j = i + 1; j < M; ++j){
					if(!equal(_copy[j][i], 0.0)){
						_copy.add_str(j, i);
						found = true;
						break;
					}
				}
			}
			
			if(!found){
				return 0.0;
			}

			for(size_t j = i + 1; j < M; ++j){
				_copy.add_str(i, j, -_copy[j][i] / _copy[i][i]);
			}
		}
		for(size_t i = 0; i < M; ++i){
			res *= _copy[i][i];
		}
		return res;
	}

	size_t rank() const{
		if constexpr(M <= N){
			Matrix<M, N, Field> _copy(m);
			for(size_t i = 0; i < M; ++i){
				if(equal(_copy[i][i], 0.0)){
					for(size_t j = i + 1; j < M; ++j){
						if(!equal(_copy[j][i], 0.0)){
							_copy.add_str(j, i);
							break;
						}
					}
				}

				if(equal(_copy[i][i], 0.0)){
					for(size_t j = i + 1; j < N; ++j){
						if(!equal(_copy[i][j], 0.0)){
							_copy.add_row(j, i);
							break;
						}
					}
				}

				if(equal(_copy[i][i], 0.0)){
					continue;
				}

				for(size_t j = i + 1; j < M; ++j){
					_copy.add_str(i, j, -_copy[j][i] / _copy[i][i]);
				}

				for(size_t j = i + 1; j < N; ++j){
					_copy.add_row(i, j, -_copy[i][j] / _copy[i][i]);
				}
			}
			size_t num = 0;
			for(size_t i = 0; i < M; ++i){
				num += !equal(_copy[i][i], 0.0);
			}
			return num;

		} else {
			return (this->transpose()).rank();
		}
	}


	Field eps() const{
		return epsilon;
	}

	void eps(const Field& new_eps) {
		epsilon = new_eps;
	}

	Field sum() const{
		Field _sum = 0.0;
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < N; ++j){
				_sum += m[i][j];
			}
		}
		return _sum;
	}

	~Matrix() = default;

};

template<size_t M, size_t N, size_t K, typename Field>
Matrix<M, K, Field> operator*(const Matrix<M, N, Field>& left, const Matrix<N, K, Field>& right){

	/*
		c[i][j] = sum_k_(a[i][k] * b[k][j]);
		N x M * M x K -> N x K

	*/

	Matrix<M, K, Field> result(0);
	for(size_t i = 0; i < M; ++i){
		for(size_t j = 0; j < K; ++j){
			for(size_t k = 0; k < N; ++k){
				result[i][j] += left[i][k] * right[k][j];
			}
		}
	}
	return result;
}


template<size_t M, size_t N, typename Field>
std::ostream& operator<<(std::ostream& out, const Matrix<M, N, Field>& m){
	for(auto& row: m.m){
		for(auto& elem: row){
			out << elem << " ";
		}
		out << "\n";
	}
	return out;
}


template<size_t M, size_t N, typename Field>
std::istream& operator>>(std::istream& in, Matrix<M, N, Field>& m){
	for(auto& row: m.m){
		for(auto& elem: row){
			in >> elem;
		}
	}
	return in;
}


template<size_t M, size_t N, typename Field>
auto operator+(const Matrix<M, N, Field>& left, const Matrix<M, N, Field>& right){
	auto _copy = left;
	_copy += right;
	return _copy;
}

template<size_t M, size_t N, typename Field>
auto operator-(const Matrix<M, N, Field>& left, const Matrix<M, N, Field>& right){
	auto _copy = left;
	_copy -= right;
	return _copy;
}

template<size_t M, size_t N, typename Field>
auto operator==(const Matrix<M, N, Field>& left, const Matrix<M, N, Field>& right){
	Matrix<M, N, Field> res(0);
	for(size_t i = 0; i < M; ++i){
		for(size_t j = 0; j < N; ++j){
			res[i][j] = (left[i][j] == right[i][j]);
		}
	}
	return res;
}

template<size_t M, size_t N, typename Field>
auto operator!=(const Matrix<M, N, Field>& left, const Matrix<M, N, Field>& right){
	Matrix<M, N, Field> res(0);
	for(size_t i = 0; i < M; ++i){
		for(size_t j = 0; j < N; ++j){
			res[i][j] = (left[i][j] != right[i][j]);
		}
	}
	return res;
}

