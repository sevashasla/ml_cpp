#pragma once

#include "../tensor/tensor.hpp"

namespace ml::preprocessing {

template<size_t Classes, typename Field>
Matrix<bool> OneHot(const Matrix<Field>& input){
	Matrix<bool> result(input.num_rows(), Classes, 0);
	for(size_t i = 0; i < input.num_rows(); ++i){
		result[i][input[i][0]] = 1;
	}
	return result;
}

} // end of ml::preprocessing
