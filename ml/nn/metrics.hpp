#pragma once

#include "../tensor/tensor.hpp"

namespace ml::metrics {

template<typename Field>
Tensor<Field> accuracy(const Tensor<Field>& real, const Tensor<Field>& predicted_classes){
	Tensor<Field> result(1, 1);
	result[0][0] = (real == predicted_classes).sum();
	result[0][0] /= real.num_rows();
	return result;
}

} // end of ml::metrics
