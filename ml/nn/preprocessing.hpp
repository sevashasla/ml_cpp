#pragma once

#include "../tensor/tensor.hpp"

namespace ml::preprocessing {

template<size_t Classes, typename Field>
Tensor<bool> OneHot(const Tensor<Field>& input){
	Tensor<bool> result(input.num_rows(), Classes, 0);
	for(size_t i = 0; i < input.num_rows(); ++i){
		result[i][input[i][0]] = 1;
	}
	return result;
}

template<size_t Classes, typename Field>
Matrix<size_t> predict(const Matrix<Field>& input){
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

} // end of ml::preprocessing
