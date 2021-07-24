#pragma once

#include <iostream>
#include <type_traits>
#include <algorithm>
#include <cmath>
#include <memory>

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

// struct HadLayer: public std::exception{
// 	const char* ptr=nullptr;
// 	HadLayer(const char* ptr): ptr(ptr){}

// 	HadLayer(const HadLayer&) = default;
// 	HadLayer(HadLayer&&) = default;
// 	HadLayer& operator=(const HadLayer&) = default;
// 	HadLayer& operator=(HadLayer&&) = default;

// 	const char* what() const noexcept override{
// 		return ptr;
// 	}
// }

class Multiplier;
class Adder;

template<typename Field=double>
class Matrix{
	template<typename FField>
	friend std::ostream& operator<<(std::ostream& out, const Matrix<FField>& m);

	template<typename FField>
	friend std::istream& operator>>(std::istream& out, Matrix<FField>& m);
	friend Multiplier;
	friend Adder;


	class Layer: public std::enable_shared_from_this<Layer>{
	public:
		Matrix<double>* res_ptr=nullptr;
		virtual void backward(const Matrix<double>&) = 0;
		virtual Matrix<double>& forward(Matrix<double>&, Matrix<double>&) = 0;
		virtual void change_res_ptr(Matrix<double>*) = 0;
		virtual ~Layer(){
			cout << "~Layer()\n";
		}
	};


//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
public:	
	std::shared_ptr<Layer> layer_ptr;
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


private:
	size_t __m;
	size_t __n;
	Field epsilon = 1e-3;
	std::vector<std::vector<Field>> matrix;

	Matrix<Field>* grad_ptr=nullptr;

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
	Matrix(): Matrix(1, 1, nullptr){}


	Matrix(size_t m, size_t n, std::shared_ptr<Layer> layer_ptr=nullptr): __m(m), __n(n), matrix(__m, std::vector<Field>(__n, 0)), layer_ptr(layer_ptr){
		for(size_t i = 0; i < std::min(__m, __n); ++i){
			matrix[i][i] = 1;
		}
	}

	Matrix(const std::pair<size_t, size_t>& sz, std::shared_ptr<Layer> layer_ptr=nullptr): Matrix(sz.first, sz.second, layer_ptr){}
	Matrix(const std::pair<size_t, size_t>& sz, size_t n, std::shared_ptr<Layer> layer_ptr=nullptr): Matrix(sz.first, sz.second, n, layer_ptr){}

	static Matrix<Field> eye(size_t m, size_t n, std::shared_ptr<Layer> layer_ptr=nullptr){
		return Matrix<Field>(m, n, layer_ptr);
	}

	static Matrix<Field> random(size_t m, size_t n, std::shared_ptr<Layer> layer_ptr=nullptr){
		Matrix<Field> res(m, n, 0, layer_ptr);
		for(size_t i = 0; i < m; ++i){
			for(size_t j = 0; j < n; ++j){
				res[i][j] = static_cast<Field>(rand()) / RAND_MAX * 2.0 - 1.0;
			}
		}
		return res;
	}

	void push_back(const std::vector<double>& v){
		matrix.push_back(v);
		++__m;
	}


	Matrix(size_t m, size_t n, const Field& f, std::shared_ptr<Layer> layer_ptr=nullptr): __m(m), __n(n), 
	matrix(__m, std::vector<Field>(__n, f)), layer_ptr(layer_ptr){}


	Matrix(const Matrix& other): __m(other.__m), __n(other.__n), 
	epsilon(epsilon), matrix(other.matrix), layer_ptr(other.layer_ptr), grad_ptr(other.grad_ptr){}


	Matrix(Matrix&& other): __m(other.__m), __n(other.__n), 
	epsilon(epsilon), matrix(std::move(other.matrix)), layer_ptr(std::move(other.layer_ptr)), grad_ptr(other.grad_ptr){
		other.grad_ptr = nullptr;
		//Maybe not good!!!
		__m = 0;
		__n = 0;
	}

	Matrix(const std::vector<std::vector<Field>>& matrix_other): __m(matrix_other.size()), 
	__n(matrix_other[0].size()), matrix(matrix_other){}

	Matrix(std::vector<std::vector<Field>>&& matrix_other): __m(matrix_other.size()), 
	__n(matrix_other[0].size()), matrix(std::move(matrix_other)){}

	Matrix& operator=(const Matrix& other) & {
		__m = other.__m;
		__n = other.__n;
		matrix = other.matrix;
		layer_ptr = other.layer_ptr;
		return *this;
	}

	Matrix& operator=(Matrix&& other) & {
		__m = other.__m;
		__n = other.__n;
		matrix = std::move(other.matrix);
		layer_ptr = std::move(other.layer_ptr);
		other.__m = other.__n = 0;
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

	void backward(const Matrix<Field>& grad_other) {
		if(layer_ptr){
			if(!grad_ptr){
				grad_ptr = new Matrix<Field>(grad_other.size(), 0.0);
			}
			
			//	I need to do change_res_ptr due to 
			//	occurs calls of copy constructors and original
			//	Matrix has died
			layer_ptr->change_res_ptr(this);

			//	call backward of the layer. It knows how to count grad
			layer_ptr->backward(grad_other);

			//	and here I need to delete layer, because on every iteration
			//	I will create graph again
			layer_ptr.reset();

		} else {
			if(!grad_ptr){

				//It means, that this is the leaf
				grad_ptr = new Matrix<Field>(grad_other);	
			}
		}
	}

	void zero_grad(){
		grad_ptr->make_zero();//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		//and call zero_grad of other
	}

	const Matrix<Field>& get_grad() const{
		return *grad_ptr;
	}

	~Matrix(){
		cout << "~Matrix()\n";
		delete grad_ptr;
	}
};


//May be template?
class Multiplier: public Matrix<double>::Layer{
/*
	A: M x N
	B: N x K
	C: M x K = A * B
	
	C_i = (A_{i 1} * B{1 1} + ... + A_{i N} * B_{N 1}, ... , A_{i 1} * B{1 K} + ... + A_{i N} * B{N K})

	df/dA_{1 1} = 1/K * (df/dC_{1 1} * B_{1 1} + ... + df/dC_{1 K} * B_{K 1})
	df/dB_{1 1} = 1/M * (df/dC_{1 1} * A_{1 1} + ... + df/dC_{M 1} * A_{M 1})

			(df/dC_{1 1}, ... , df/dC_{1 K})
	grad_f = 				...
			(df/dC_{M 1}, ... , df/dC_{M K})
	
	grad_A = 1/K * grad_f * B^T
	grad_B = 1/M * A^T * grad_f
*/

private:
	Matrix<double>* left_ptr;
	Matrix<double>* right_ptr;
	Matrix<double>* res_ptr;

	virtual void change_res_ptr(Matrix<double>* ptr) {
		res_ptr = ptr;
	}

	Matrix<double> matmul(const Matrix<double>& left, const Matrix<double>& right) const{
	/*
		c[i][j] = sum_k_(a[i][k] * b[k][j]);
		(M x N) * (N x K) -> (M x K)
	*/

		size_t M = left.num_rows();
		size_t N = left.num_columns();
		size_t K = right.num_columns();


		Matrix<double> result(M, K, 0, nullptr);
		for(size_t i = 0; i < M; ++i){
			for(size_t j = 0; j < K; ++j){
				for(size_t k = 0; k < N; ++k){
					result[i][j] += left[i][k] * right[k][j];
				}
			}
		}

		return result;
	}


	Matrix<double> mulscalar(double scalar, const Matrix<double>& matrix) const{
		Matrix<double> result = matrix;
		for(size_t i = 0; i < result.num_rows(); ++i){
			for(size_t j = 0; j < result.num_columns(); ++j){
				result[i][j] *= scalar;
			}
		}
		return result;
	}

public:
	Multiplier(){
		res_ptr = new Matrix<double>(1, 1, 0.0);
	}

	Matrix<double>& forward(Matrix<double>& left, Matrix<double>& right) override{
		if(left.num_columns() != right.num_rows()){
			delete res_ptr;
			throw BadShape("sizes must be m x n * n x k. Multiplier, forward");
		}

		left_ptr = &left;
		right_ptr = &right;
		*res_ptr = matmul(left, right);

		res_ptr->layer_ptr = shared_from_this();
		return *res_ptr;
	}


	void backward(const Matrix<double>& grad_other) override{
		auto& grad_current = *(res_ptr->grad_ptr);
		grad_current += grad_other;

		left_ptr->backward(mulscalar(1.0 / right_ptr->num_columns(), matmul(grad_current, right_ptr->transpose())));
		right_ptr->backward(mulscalar(1.0 / left_ptr->num_rows(), matmul(left_ptr->transpose(), grad_current)));
	}

	~Multiplier(){
		cout << "~Multiplier()\n";
		delete res_ptr;
	}
};



class Adder: public Matrix<double>::Layer{
/*
	A + B = C
	y = f(C)
			(dy/dC_{1 1}, ... , dy/dC_{1 N})
	grad_y = 			...
			(dy/dC_{M 1}, ... , dy/dC_{M N})
	

			(dy/dA_{1 1}, ... , dy/dA_{1 N})	(dy/dC_{1 1}, ... , dy/dC_{1 N})
	grad_A = 			...					=				...
			(dy/dA_{M 1}, ... , dy/dA_{M N})	(dy/dC_{1 1}, ... , dy/dC_{1 N})
				
	similarly for grad_B
*/
private:
	Matrix<double>* left_ptr=nullptr;
	Matrix<double>* right_ptr=nullptr;
	Matrix<double>* res_ptr=nullptr;

	virtual void change_res_ptr(Matrix<double>* ptr){
		res_ptr = ptr; 
	}

public:
	Adder(const Adder&) = delete;
	Adder& operator=(const Adder&) = delete;

	Adder(){
		res_ptr = new Matrix<double>(1, 1, 0.0);
	}

	Matrix<double>& forward(Matrix<double>& left, Matrix<double>& right) override{
		if(left.size() != right.size()){
			delete res_ptr;
			throw BadShape("Wrong shapes of matrices. Adder, forward");
		}

		this->left_ptr = &left;
		this->right_ptr = &right;


///
		*res_ptr = left;
		res_ptr->layer_ptr = shared_from_this();
		
///

		for(size_t i = 0; i < left.num_rows(); ++i){
			for(size_t j = 0; j < left.num_columns(); ++j){
				(*res_ptr)[i][j] += right[i][j];
			}
		}
		return *res_ptr;
	}

	void backward(const Matrix<double>& grad_other) override{
		auto& grad_current = *(res_ptr->grad_ptr);
		grad_current += grad_other;
		left_ptr->backward(grad_current);
		right_ptr->backward(grad_current);
	}

	~Adder(){
		cout << "~Adder()\n";
		delete res_ptr;
	}
};


// class Transposer: public Matrix<double>::Layer{
// private:

// public:
// };



template<typename Field>
Matrix<Field> operator*(Matrix<Field>& left, Matrix<Field>& right){
	std::shared_ptr<Multiplier> multiplier_ptr = std::make_shared<Multiplier>();
	return multiplier_ptr->forward(left, right);
}




template<typename Field>
Matrix<Field> operator+(Matrix<Field>& left, Matrix<Field>& right){
	std::shared_ptr<Adder> adder_ptr = std::make_shared<Adder>();
	return adder_ptr->forward(left, right);
}

//wrong!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
template<typename Field>
auto operator-(Matrix<Field>& left, Matrix<Field>& right){
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

