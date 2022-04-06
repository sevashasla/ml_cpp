#pragma once

#include <iostream>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

struct BadShape: public std::exception{
	const char* ptr=nullptr;
	BadShape(const char* ptr): ptr(ptr){}

	BadShape(const BadShape&) = default;
	BadShape(BadShape&&) = default;
	BadShape& operator=(const BadShape&) = default;
	BadShape& operator=(BadShape&&) = default;

	const char* what() const noexcept override{
		return ptr;
	}
};

template<typename T>
bool equal(const T& left, const T& right, double epsilon=1e-3) {
	return (-epsilon <= right - left) && (right - left <= epsilon);
}

template<typename Field=double>
class Matrix{
	template<typename FField>
	friend std::ostream& operator<<(std::ostream& out, const Matrix<FField>& m);

	template<typename FField>
	friend std::istream& operator>>(std::istream& out, Matrix<FField>& m);


private:
	// second = first * coeff + second
	void add_str(size_t first, size_t second, Field coeff = 1){
		for(size_t k = 0; k < n_; ++k){
			matrix[second][k] += coeff * matrix[first][k];
		}
	}

	void add_row(size_t first, size_t second, Field coeff = 1){
		for(size_t k = 0; k < m_; ++k){
			matrix[k][second] += coeff * matrix[k][first];
		}
	}

protected:
	size_t m_;
	size_t n_;
	std::vector<std::vector<Field>> matrix;

public:
	Matrix(): Matrix(0, 0){}

	Matrix(const Field& f): Matrix(1, 1, f){}

	//	Construct Matrix: m x n with  1 on the main diag
	Matrix(size_t m, size_t n): m_(m), n_(n), 
	matrix(m_, std::vector<Field>(n_, 0)){
		for(size_t i = 0; i < std::min(m_, n_); ++i){
			matrix[i][i] = 1;
		}
	}

	Matrix(std::pair<size_t, size_t> size): Matrix(size.first, size.second) {}
	Matrix(std::pair<size_t, size_t> size, const Field& f): Matrix(size.first, size.second, f) {}

	Matrix(size_t m, size_t n, const Field& f): m_(m), n_(n), 
		matrix(m_, std::vector<Field>(n_, f)){}

	Matrix(const Matrix& other) = default;
	Matrix(Matrix&& other) = default;
	Matrix& operator=(const Matrix& other) & = default;
	Matrix& operator=(Matrix&& other) & = default;

	//	Construct Matrix from vector
	Matrix(const std::vector<std::vector<Field>>& matrix_other): m_(matrix_other.size()), 
	n_(matrix_other[0].size()), matrix(matrix_other){}

	Matrix(std::vector<std::vector<Field>>&& matrix_other): m_(matrix_other.size()), 
	n_(matrix_other[0].size()), matrix(std::move(matrix_other)){}


	//	Get Matrix: m x n with 1 on the main diag
	static Matrix<Field> eye(size_t m, size_t n){
		return Matrix<Field>(m, n);
	}

	//	Get Matrix: m x n with containing random numbers
	static Matrix<Field> random(size_t m, size_t n){
		Matrix<Field> res(m, n, 0);
		for(size_t i = 0; i < m; ++i){
			for(size_t j = 0; j < n; ++j){
				res[i][j] = static_cast<Field>(rand()) / RAND_MAX * 2.0 - 1.0;
			}
		}
		return res;
	}

	//	For push another one row
	void push_row(const std::vector<double>& v){
		matrix.push_back(v);
		++m_;
	}

	//	For push another one column
	void push_column(const std::vector<double>& v){
		for(size_t i = 0; i < m_; ++i){
			matrix[i].push_back(v[i]);
		}
		++n_;
	}

	std::vector<Field>& operator[](size_t i) {
		return matrix[i];
	}

	const std::vector<Field>& operator[](size_t i) const {
		return matrix[i];
	}

	Matrix& operator*=(const Field& f){
		for(auto& row: matrix){
			for(auto& elem: row){
				elem *= f;
			}
		}
		return *this;
	}

	Matrix& operator/=(const Field& f){
		for(auto& row: matrix){
			for(auto& elem: row){
				elem /= f;
			}
		}
		return *this;
	}

	Matrix& operator/=(const Matrix& other) {
		if (size() != other.size()) {
			throw BadShape("Wrong shapes of matrices. Matrix, /");
		}
		for(size_t i = 0; i < m_; ++i) {
			for(size_t j = 0; j < n_; ++j) {
				matrix[i][j] /= other[i][j];
			}
		}
		return *this;
	}

	Matrix& operator*=(const Matrix<Field>& other){
		auto transposed = other.transpose();
		//m x n * n x k
		if (n_ != other.m_){
			throw BadShape("Wrong shapes of matrices. Matrix, *");
		}

		Matrix<Field> result(m_, other.n_, 0);
		for(size_t i = 0; i < m_; ++i){
			for(size_t j = 0; j < other.n_; ++j){
				for(size_t k = 0; k < n_; ++k){
					result[i][j] += matrix[i][k] * transposed[j][k];
				}
			}
		}

		*this = std::move(result);
		return *this;
	}

	Matrix& operator+=(const Matrix<Field>& other){
		if(size() != other.size()){
			throw BadShape("Wrong shapes of matrices. Matrix, +");
		}

		for(size_t i = 0; i < m_; ++i){
			for(size_t j = 0; j < n_; ++j){
				matrix[i][j] += other[i][j];
			}
		}
		return *this;
	}

	Matrix& operator+=(const Field& field){
		for(size_t i = 0; i < m_; ++i){
			for(size_t j = 0; j < n_; ++j){
				matrix[i][j] += field;
			}
		}
		return *this;
	}

	Matrix& operator-=(const Matrix<Field>& other){
		if(size() != other.size()){
			throw BadShape("Wrong shapes of matrices. Matrix, -");
		}

		for(size_t i = 0; i < m_; ++i){
			for(size_t j = 0; j < n_; ++j){
				matrix[i][j] -= other[i][j];
			}
		}
		return *this;
	}

	//	make current matrix full of 0
	void make_zero(){
		for(size_t i = 0; i < m_; ++i){
			for(size_t j = 0; j < n_; ++j){
				matrix[i][j] = 0.0;
			}
		}
	}

	Matrix abs() const {
		Matrix copy(*this);
		for(size_t i = 0; i < m_; ++i){
			for(size_t j = 0; j < n_; ++j){
				copy[i][j] = std::abs(copy[i][j]);
			}
		}
		return copy;
	}

	Matrix transpose() const{
		Matrix<Field> transposed(n_, m_);
		for(size_t i = 0; i < n_; ++i){
			for(size_t j = 0; j < m_; ++j){
				transposed[i][j] = matrix[j][i];
			}
		}
		return transposed;
	}

	//	Get result if we sum every element
	Field sum() const{
		Field _sum = 0;
		for(auto& row: matrix){
			for(auto& elem: row){
				_sum += elem;
			}
		}
		return _sum;
	}


	std::pair<size_t, size_t> size() const{
		return std::make_pair(m_, n_);
	}

	size_t num_rows() const{
		return m_;
	}

	size_t num_columns() const{
		return n_;
	}

	template<typename FField>
	explicit operator Matrix<FField>() const{
		Matrix<FField> res(m_, n_, 0);
		for(size_t i = 0; i < m_; ++i){
			for(size_t j = 0; j < n_; ++j){
				res[i][j] = static_cast<FField>(matrix[i][j]);
			}
		}
		return res;
	}

	~Matrix() = default;
};

template<typename Field>
Matrix<Field> operator+(Matrix<Field> left, const Matrix<Field>& right){
	return left += right;
}

template<typename Field>
Matrix<Field> operator+(Matrix<Field> left, const Field& field){
	return left += field;
}

template<typename Field>
Matrix<Field> operator*(Matrix<Field> left, const Field& field){
	return left *= field;
}

template<typename Field>
Matrix<Field> operator-(Matrix<Field> left, const Matrix<Field>& right){
	return left -= right;
}

template<typename Field>
Matrix<Field> operator*(Matrix<Field> left, const Matrix<Field>& right){
	return left *= right;
}

template<typename Field>
Matrix<Field> operator/(Matrix<Field> left, const Matrix<Field>& right){
	return left /= right;
}

template<typename Field>
Matrix<Field> operator==(const Matrix<Field>& left, const Matrix<Field>& right){	
	if(left.size() != right.size()){
		throw BadShape("One can't compare (==) matrices with different shapes");
	}

	Matrix<Field> res(left.num_rows(), left.num_columns(), 0);
	for(size_t i = 0; i < left.num_rows(); ++i){
		for(size_t j = 0; j < left.num_columns(); ++j){
			res[i][j] = equal(left[i][j], right[i][j]);
		}
	}
	return res;
}

template<typename Field>
Matrix<Field> operator!=(const Matrix<Field>& left, const Matrix<Field>& right){
	if(left.size() != right.size()){
		throw BadShape("One can't compare (!=) matrices with different shapes");
	}

	Matrix<Field> res(left.num_rows(), left.num_columns(), 0);
	for(size_t i = 0; i < left.num_rows(); ++i){
		for(size_t j = 0; j < left.num_columns(); ++j){
			res[i][j] = !equal(left[i][j], right[i][j]);
		}
	}
	return res;
}

template<typename Field>
std::ostream& operator<<(std::ostream& out, const Matrix<Field>& m){
	for(auto& row: m.matrix){
		for(auto& elem: row){
			out << elem << " ";
		}
		out << "\n";
	}
	return out;
}


template<typename Field>
std::istream& operator>>(std::istream& in, Matrix<Field>& m){
	for(auto& row: m.matrix){
		for(auto& elem: row){
			in >> elem;
		}
	}
	return in;
}
