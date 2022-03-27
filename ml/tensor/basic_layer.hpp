#pragma once

#include "../matrix/matrix.hpp"
#include "tensor.hpp"

namespace ml{

template<typename Field>
class Tensor;

template<typename Field>
class Layer: public std::enable_shared_from_this<Layer<Field>> {

template<typename FField>
friend class Tensor;

public:
    virtual ~Layer() = default;

    // forward for one argument
    virtual Tensor<Field> forward(Tensor<Field>&){
        throw std::runtime_error("This Layer expected 2 arguments");
    }

    // forward for two arguments
    virtual Tensor<Field> forward(Tensor<Field>&, Tensor<Field>&){
        throw std::runtime_error("This Layer expected 1 argument");
    }

private:
    virtual void backward_(const Matrix<Field>&) = 0;
    virtual void make_step_(Field step) = 0;
    virtual void break_graph_() = 0;
    virtual void zero_grad_() = 0;
};

} // end of namespace ml
