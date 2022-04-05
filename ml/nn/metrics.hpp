#pragma once

#include "../tensor/tensor.hpp"

namespace ml::metrics {

template<typename Field, typename T>
Matrix<Field> accuracy(const Matrix<T>& real, const Matrix<T>& predicted_classes){
	Matrix<Field> result(1, 1);
	result[0][0] = (real == predicted_classes).sum();
	result[0][0] /= real.num_rows();
	return result;
}

template<typename Field>
Matrix<Field> meanAveragePercentageError(const Matrix<Field>& real, const Matrix<Field>& predicted_classes){
	Matrix<Field> result(1, 1);
	result[0][0] = ((real - predicted_classes).abs() / (real + 1e-9)).sum();
	return result;	
}

} // end of ml::metrics
