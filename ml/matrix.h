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


class Multiplier;
class Adder;

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

public: 
	//	common class for every layer for neural network
	//	It's good to make it outside Matrix, but I can't
	//	due to not full implemented class

	class Layer: public std::enable_shared_from_this<Layer>{
	public:
		Matrix<double>* res_ptr=nullptr;

		//	For pushing gradient deeper
		virtual void backward(const Matrix<double>&) = 0;

		//	I have different methods with different number of input arguments
		virtual Matrix<double> forward(Matrix<double>&, Matrix<double>&){
			throw std::runtime_error("One tries to call forward(Matrix<double>&, Matrix<double>&)");
			return Matrix<double>();
		}
		virtual Matrix<double> forward(Matrix<double>&){
			throw std::runtime_error("One tries to call forward(Matrix<double>&)");
			return Matrix<double>();
		}

		//	For change weights
		virtual void make_step(double) = 0;
		virtual void zero_grad() = 0;

		//	There are no virtual variables, but I need to change
		//	variable of the lower class
		virtual void change_res_ptr(Matrix<double>*) = 0;

		//	For breaking graph because on every iteration it has to been builded again
		virtual void break_graph() = 0;
		virtual ~Layer() = default;
	};


private:
	size_t __m;
	size_t __n;
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


public:

	//	This pointer shows how it was created
	std::shared_ptr<Layer> layer_ptr;
	std::shared_ptr<Matrix<Field>> grad_ptr;

	Matrix(): Matrix(0, 0, nullptr){}

	//	Construct Matrix: m x n with layer_ptr and with 1 on the main diag
	Matrix(size_t m, size_t n, std::shared_ptr<Layer> layer_ptr=nullptr): __m(m), __n(n), 
	matrix(__m, std::vector<Field>(__n, 0)), layer_ptr(layer_ptr){
		for(size_t i = 0; i < std::min(__m, __n); ++i){
			matrix[i][i] = 1;
		}
	}

	//	Construct Matrix: sz with layer_ptr
	Matrix(const std::pair<size_t, size_t>& sz, std::shared_ptr<Layer> layer_ptr=nullptr): Matrix(sz.first, sz.second, layer_ptr){}

	//	Construct Matrix: sz with layer_ptr and full of "n"
	Matrix(const std::pair<size_t, size_t>& sz, size_t n, std::shared_ptr<Layer> layer_ptr=nullptr): Matrix(sz.first, sz.second, n, layer_ptr){}

	//	Construct Matrix: m x n, full of f
	Matrix(size_t m, size_t n, const Field& f, std::shared_ptr<Layer> layer_ptr=nullptr): __m(m), __n(n), 
	matrix(__m, std::vector<Field>(__n, f)), layer_ptr(layer_ptr){}

	Matrix(const Matrix& other): __m(other.__m), __n(other.__n), 
	matrix(other.matrix), layer_ptr(other.layer_ptr), grad_ptr(other.grad_ptr){}

	Matrix(Matrix&& other): __m(other.__m), __n(other.__n), 
	matrix(std::move(other.matrix)), layer_ptr(std::move(other.layer_ptr)), grad_ptr(std::move(other.grad_ptr)){
		other.__m = other.__n = 0;
	}

	//	Construct Matrix from vector
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
		grad_ptr = std::move(other.grad_ptr);
		other.__m = other.__n = 0;
		return *this;
	}

	//	Get Matrix: m x n with layer_ptr and with 1 on the main diag
	static Matrix<Field> eye(size_t m, size_t n, std::shared_ptr<Layer> layer_ptr=nullptr){
		return Matrix<Field>(m, n, layer_ptr);
	}

	//	Get Matrix: m x n with layer_ptr and it contains random numbers
	static Matrix<Field> random(size_t m, size_t n, std::shared_ptr<Layer> layer_ptr=nullptr){
		Matrix<Field> res(m, n, 0, layer_ptr);
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

	void backward(const Matrix<Field>& grad_other){
		//	I can't create grad_ptr in constructors 
		//	otherwise there is infinite recursion
		if(!grad_ptr){
			grad_ptr = std::make_shared<Matrix<Field>>(grad_other.size(), 0.0);
		}

		//	the most important step
		*grad_ptr += grad_other;

		if(layer_ptr){	
			//	I need to do change_res_ptr due to 
			//	occurs calls of copy constructors and I 
			//	don't have original matrix
			layer_ptr->change_res_ptr(this);

			//	call backward of the layer. It knows how to count grad
			layer_ptr->backward(grad_other);
		}
	}

	//	I need this one for loss functions
	void backward(){
		if(!grad_ptr){
			grad_ptr = std::make_shared<Matrix<Field>>();
		}
		if(layer_ptr){
			layer_ptr->change_res_ptr(this);			
			layer_ptr->backward();
		}
	}

	void make_step(double step){
		//	Here I call layer, because it knows how to change weights
		if(layer_ptr){
			layer_ptr->make_step(step);
		}
	}

	void break_graph(){
		//	with help of recursion I will break the graph
		if(layer_ptr){
			layer_ptr->break_graph();
			layer_ptr.reset();
		}
		//	I can't do it in another order
		//	because I destory connection between them
	}

	void zero_grad(){
		//	make own grad equal to zero and push instruction down
		grad_ptr->make_zero();
		if(layer_ptr){
			layer_ptr->zero_grad();
		}
	}

	const Matrix<Field>& get_grad() const{
		return *grad_ptr;
	}

	//	I don't allocate vatiables
	//	So there are trivial destructor
	~Matrix() = default;
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


public:
	Multiplier(){}

	static Matrix<double> matmul(const Matrix<double>& left, const Matrix<double>& right){
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

	static Matrix<double> mulscalar(double scalar, const Matrix<double>& matrix){
		Matrix<double> result = matrix;
		for(size_t i = 0; i < result.num_rows(); ++i){
			for(size_t j = 0; j < result.num_columns(); ++j){
				result[i][j] *= scalar;
			}
		}
		return result;
	}

	Matrix<double> forward(Matrix<double>& left, Matrix<double>& right) override{
		if(left.num_columns() != right.num_rows()){
			throw BadShape("sizes must be m x n * n x k. Multiplier, forward");
		}

		left_ptr = &left;
		right_ptr = &right;
		Matrix<double> result = matmul(left, right);
		result.layer_ptr = shared_from_this();
		return result;
	}

	void backward(const Matrix<double>& grad_other) override{
		auto& grad_current = *(res_ptr->grad_ptr);

		left_ptr->backward(mulscalar(1.0 / right_ptr->num_columns(), matmul(grad_current, right_ptr->transpose())));
		right_ptr->backward(mulscalar(1.0 / left_ptr->num_rows(), matmul(left_ptr->transpose(), grad_current)));
	}

	void zero_grad(){
		left_ptr->zero_grad();
		right_ptr->zero_grad();
	}

	void make_step(double step) override{
		left_ptr->make_step(step);
		right_ptr->make_step(step);
	}

	void break_graph(){
		left_ptr->break_graph();
		right_ptr->break_graph();
	}

	~Multiplier() = default;
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

	static Matrix<double> add(Matrix<double>& left, Matrix<double>& right){
		Matrix<double> result = left;
		for(size_t i = 0; i < left.num_rows(); ++i){
			for(size_t j = 0; j < left.num_columns(); ++j){
				result[i][j] += right[i][j];
			}
		}
		return result;		
	}

public:
	Adder(const Adder&) = delete;
	Adder& operator=(const Adder&) = delete;

	Adder(){}

	Matrix<double> forward(Matrix<double>& left, Matrix<double>& right) override{
		if(left.size() != right.size()){
			throw BadShape("Wrong shapes of matrices. Adder, forward");
		}

		left_ptr = &left;
		right_ptr = &right;
		
		Matrix<double> result = add(left, right);
		result.layer_ptr = shared_from_this();

		return result;
	}

	void backward(const Matrix<double>& grad_other) override{
		auto& grad_current = *(res_ptr->grad_ptr);

		left_ptr->backward(grad_current);
		right_ptr->backward(grad_current);
	}

	void zero_grad(){
		left_ptr->zero_grad();
		right_ptr->zero_grad();
	}

	void make_step(double step) override{
		left_ptr->make_step(step);
		right_ptr->make_step(step);
	}

	void break_graph(){
		left_ptr->break_graph();
		right_ptr->break_graph();
	}

	~Adder() = default;
};


class Subtractor: public Matrix<double>::Layer{
private:
	Matrix<double>* res_ptr;
	Matrix<double>* left_ptr;
	Matrix<double>* right_ptr;

	static Matrix<double> subtract(const Matrix<double>& left, const Matrix<double>& right){
		Matrix<double> result(left);
		for(size_t i = 0; i < left.num_rows(); ++i){
			for(size_t j = 0; j < left.num_columns(); ++j){
				result[i][j] -= right[i][j];
			}
		}
		return result;
	}

public:
	void change_res_ptr(Matrix<double>* ptr) override{
		res_ptr = ptr;
	}

	Matrix<double> forward(Matrix<double>& left, Matrix<double>& right) override{
		if(left.size() != right.size()){
			throw BadShape("Wrong shapes of matrices. Subtractor, forward");
		}

		left_ptr = &left;
		right_ptr = &right;

		Matrix<double> result = subtract(left, right);
		result.layer_ptr = shared_from_this();

		return result;
	}

	void backward(const Matrix<double>& grad_other) override{
		auto& grad_current = *(res_ptr->grad_ptr);
		left_ptr->backward(grad_current);
		right_ptr->backward(Multiplier::mulscalar(-1, grad_current));
	}

	void zero_grad() override{
		left_ptr->zero_grad();
		right_ptr->zero_grad();
	}

	void make_step(double step) override{
		left_ptr->make_step(step);
		right_ptr->make_step(step);
	}

	void break_graph() override{
		left_ptr->break_graph();
		right_ptr->break_graph();
	}
};


class Transposer: public Matrix<double>::Layer{
private:
	Matrix<double>* res_ptr;
	Matrix<double>* input_ptr;

public:
	Transposer() = default;

	static Matrix<double> transpose(const Matrix<double>& input){
		Matrix<double> result(input.num_columns(), input.num_rows());
		for(size_t i = 0; i < input.num_columns(); ++i){
			for(size_t j = 0; j < input.num_rows(); ++j){
				result[i][j] = input[j][i];
			}
		}
		return result;
	}

	Matrix<double> forward(Matrix<double>& input) override {
		input_ptr = &input;
		Matrix<double> result = transpose(input);
		result.layer_ptr = shared_from_this();
		return result;
	}


	void backward(const Matrix<double>& grad_other) override {
		auto& grad_current = *(res_ptr->grad_ptr);
		input_ptr->backward(transpose(grad_current));
	}

	void zero_grad() override {
		input_ptr->zero_grad();
	}

	void make_step(double step) override {
		input_ptr->make_step(step);
	}

	void break_graph() override {
		input_ptr->break_graph();
	}
};



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


template<typename Field>
Matrix<Field> operator-(Matrix<Field>& left, Matrix<Field>& right){
	std::shared_ptr<Subtractor> subtractor_ptr = std::make_shared<Subtractor>();
	return subtractor_ptr->forward(left, right);
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

