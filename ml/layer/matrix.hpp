#pragma once

#include <iostream>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using std::cout;
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

protected:
	std::vector<std::vector<Field>> matrix;

private:
	size_t __m;
	size_t __n;

	// second = first * coeff + second
	void add_str(size_t first, size_t second, Field coeff = 1){
		for(size_t k = 0; k < __n; ++k){
			matrix[second][k] += coeff * matrix[first][k];
		}
	}

	void add_row(size_t first, size_t second, Field coeff = 1){
		for(size_t k = 0; k < __m; ++k){
			matrix[k][second] += coeff * matrix[k][first];
		}
	}

public:
	Matrix(): Matrix(0, 0, nullptr){}

	//	Construct Matrix: m x n with  1 on the main diag
	Matrix(size_t m, size_t n): __m(m), __n(n), 
	matrix(__m, std::vector<Field>(__n, 0)){
		for(size_t i = 0; i < std::min(__m, __n); ++i){
			matrix[i][i] = 1;
		}
	}

	//	Construct Matrix: m x n, full of f
	Matrix(size_t m, size_t n, const Field& f): __m(m), __n(n), 
	matrix(__m, std::vector<Field>(__n, f)){}

	Matrix(const Matrix& other): __m(other.__m), __n(other.__n), 
	matrix(other.matrix){}

	Matrix(Matrix&& other): __m(other.__m), __n(other.__n), 
	matrix(std::move(other.matrix)){
		other.__m = other.__n = 0;
	}

	//	Construct Matrix from vector
	Matrix(const std::vector<std::vector<Field>>& matrix_other): __m(matrix_other.size()), 
	__n(matrix_other[0].size()), matrix(matrix_other){}

	Matrix(std::vector<std::vector<Field>>&& matrix_other): __m(matrix_other.size()), 
	__n(matrix_other[0].size()), matrix(std::move(matrix_other)){}

	Matrix& operator=(const Matrix& other) & = default;
	Matrix& operator=(Matrix&& other) & = default;

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

	//	For adding another one row
	void push_row(const std::vector<double>& v){
		matrix.push_back(v);
		++__m;
	}

	//	For adding another one column
	void push_column(const std::vector<double>& v){
		for(size_t i = 0; i < __m; ++i){
			matrix[i].push_back(v[i]);
		}
		++__n;
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

	Matrix& operator*=(const Matrix<Field>& other){
		//m x n * n x k
		if(__n != other.__m){
			throw BadShape("Wrong shapes of matrices. Matrix, *=");
		}

		Matrix<Field> result(__m, other.__n, 0);
		for(size_t i = 0; i < __m; ++i){
			for(size_t j = 0; j < other.__n; ++j){
				for(size_t k = 0; k < __n; ++k){
					result[i][j] += matrix[i][k] * other[k][j];
				}
			}
		}

		*this = std::move(result);
		return *this;
	}

	Matrix& operator+=(const Matrix<Field>& other){
		if(size() != other.size()){
			throw BadShape("Wrong shapes of matrices. Matrix, +=");
		}

		for(size_t i = 0; i < __m; ++i){
			for(size_t j = 0; j < __n; ++j){
				matrix[i][j] += other[i][j];
			}
		}

		return *this;
	}

	Matrix& operator-=(const Matrix<Field>& other){
		if(size() != other.size()){
			throw BadShape("Wrong shapes of matrices. Matrix, -=");
		}

		for(size_t i = 0; i < __m; ++i){
			for(size_t j = 0; j < __n; ++j){
				matrix[i][j] -= other[i][j];
			}
		}
		return *this;
	}

	//	make current matrix full of 0
	void make_zero(){
		for(size_t i = 0; i < __m; ++i){
			for(size_t j = 0; j < __n; ++j){
				matrix[i][j] = 0.0;
			}
		}
	}

	Matrix<Field> transpose() const{
		Matrix<Field> transposed(__n, __m, 0, nullptr);
		for(size_t i = 0; i < __n; ++i){
			for(size_t j = 0; j < __m; ++j){
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
		return std::make_pair(__m, __n);
	}

	size_t num_rows() const{
		return __m;
	}

	size_t num_columns() const{
		return __n;
	}

	template<typename FField>
	explicit operator Matrix<FField>() const{
		Matrix<FField> res(__m, __n, 0, nullptr);
		for(size_t i = 0; i < __m; ++i){
			for(size_t j = 0; j < __n; ++j){
				res[i][j] = static_cast<FField>(matrix[i][j]);
			}
		}
		return res;
	}

	//	I don't allocate vatiables
	//	So there is trivial destructor
	~Matrix() = default;
};

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
