#pragma once

#include "../tensor/tensor.hpp"

namespace ml::postprocessing {

template<size_t Classes, typename Field>
Matrix<size_t> predict(const Matrix<Field>& input){
	if (input.num_columns() != Classes) {
		throw BadShape("num_columns != classes, predict");
	}
	Matrix<size_t> result(input.num_rows(), 1, 0);
	for(size_t num = 0; num < input.num_rows(); ++num){
		size_t index_max = 0;
		for(size_t i = 0; i < Classes; ++i){
			if(input[num][index_max] < input[num][i]){
				index_max = i;
			}
		}
		result[num][0] = index_max;
	}
	return result;
}

template<size_t Classes, typename Field>
Matrix<Field> predictProba(const Matrix<Field>& input){
	if (input.size().second != Classes) {
		throw BadShape("n != classes, predictProba");
	}

	Matrix<Field> result(input.num_rows(), Classes, 0);
	for(size_t num = 0; num < input.num_rows(); ++num){
		Field sum_ = 0;
		for(size_t i = 0; i < input.num_columns(); ++i){
			result[num][i] = std::exp(input[num][i]);
			sum_ += result[num][i];
		}
		for(size_t i = 0; i < input.num_columns(); ++i){
			result[num][i] /= sum_;
		}
	}
	return result;
}

} // end of ml::postprocessing