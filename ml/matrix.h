#include <iostream>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <initializer_list>



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


template<typename Field=double>
class Matrix{
	template<typename FField>
	friend std::ostream& operator<<(std::ostream& out, const Matrix<FField>& m);

	template<typename FField>
	friend std::istream& operator>>(std::istream& out, Matrix<FField>& m);


private:
	size_t __m;
	size_t __n;
	Field epsilon = 1e-3;
	std::vector<std::vector<Field>> matrix;

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

	bool equal(const Field& left, const Field& right) const{
		return (-epsilon <= right - left) && (right - left <= epsilon);
	}


public:
	Matrix(): Matrix(0, 0){}

	Matrix(size_t m, size_t n): __m(m), __n(n), matrix(__m, std::vector<Field>(__n, 0)){
		for(size_t i = 0; i < std::min(__m, __n); ++i){
			matrix[i][i] = 1;
		}
	}


	static Matrix<Field> eye(size_t m, size_t n){
		return Matrix<Field>(m, n);
	}

	static Matrix<Field> random(size_t m, size_t n){
		Matrix<Field> res(m, n, 0);
		for(size_t i = 0; i < m; ++i){
			for(size_t j = 0; j < n; ++j){
				res[i][j] = static_cast<Field>(rand()) / static_cast<Field>(RAND_MAX);
			}
		}
		return res;
	}

	void push_back(const std::vector<double>& v){
		matrix.push_back(v);
	}


	Matrix(size_t m, size_t n, const Field& f): __m(m), __n(n), matrix(__m, std::vector<Field>(__n, f)){}
	Matrix(const Matrix&) = default;
	Matrix(Matrix&&) = default;
	
	Matrix(const std::vector<std::vector<Field>>& matrix_other): __m(matrix_other.size()),
																 __n(matrix_other[0].size()), 
																 matrix(matrix_other){}
	Matrix(std::vector<std::vector<Field>>&& matrix_other): __m(matrix_other.size()),
															__n(matrix_other[0].size()), 
															matrix(std::move(matrix_other)){}

	Matrix& operator=(const Matrix& other) & {
		__m = other.__m;
		__n = other.__n;
		matrix = other.matrix;
		return *this;
	}

	Matrix& operator=(Matrix&& other) & {
		__m = other.__m;
		__n = other.__n;
		matrix = std::move(other.matrix);
		return *this;
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
			throw BadShape("One can't multiply (*=) matrices with such shapes");
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

		if(__n != other.__n || __m != other.__m){
			throw BadShape("One can't add (+=) matrices with such shapes");
		}

		for(size_t i = 0; i < __m; ++i){
			for(size_t j = 0; j < __n; ++j){
				matrix[i][j] += other[i][j];
			}
		}
		return *this;
	}

	Matrix& operator-=(const Matrix<Field>& other){
		
		if(__n != other.__n || __m != other.__m){
			throw BadShape("One can't subtract (-=) matrices with such shapes");
		}

		for(size_t i = 0; i < __m; ++i){
			for(size_t j = 0; j < __n; ++j){
				matrix[i][j] -= other[i][j];
			}
		}
		return *this;
	}

	Field& tr() const{
		Field res = 0;
		for(size_t i = 0; i < std::min(__m, __n); ++i){
			res += matrix[i][i];
		}
		return res;
	}

	Matrix<Field> transpose() const{
		Matrix<Field> transposed(__n, __m, 0);
		for(size_t i = 0; i < __n; ++i){
			for(size_t j = 0; j < __m; ++j){
				transposed[i][j] = matrix[j][i];
			}
		}
		return transposed;
	}


	Field eps() const{
		return epsilon;
	}

	void eps(const Field& new_eps) {
		epsilon = new_eps;
	}

	Field sum() const{
		Field _sum = 0.0;
		for(size_t i = 0; i < __m; ++i){
			for(size_t j = 0; j < __n; ++j){
				_sum += matrix[i][j];
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

	~Matrix() = default;

};

template<typename Field>
Matrix<Field> operator*(const Matrix<Field>& left, const Matrix<Field>& right){

	/*
		c[i][j] = sum_k_(a[i][k] * b[k][j]);
		(M x N) * (N x K) -> (M x K)
	*/

	if(left.num_columns() != right.num_rows()){
			throw BadShape("One can't multiply (*) matrices with such shapes");
	}

	size_t M = left.num_rows();
	size_t N = left.num_columns();
	size_t K = right.num_columns();

	Matrix<Field> result(M, K, 0);
	for(size_t i = 0; i < M; ++i){
		for(size_t j = 0; j < K; ++j){
			for(size_t k = 0; k < N; ++k){
				result[i][j] += left[i][k] * right[k][j];
			}
		}
	}
	return result;
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


template<typename Field>
auto operator+(const Matrix<Field>& left, const Matrix<Field>& right){
	auto _copy = left;
	_copy += right;
	return _copy;
}

template<typename Field>
auto operator-(const Matrix<Field>& left, const Matrix<Field>& right){
	auto _copy = left;
	_copy -= right;
	return _copy;
}

template<typename Field>
auto operator==(const Matrix<Field>& left, const Matrix<Field>& right){	
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
auto operator!=(const Matrix<Field>& left, const Matrix<Field>& right){
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

