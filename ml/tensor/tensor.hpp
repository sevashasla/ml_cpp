#pragma once

#include "../matrix/matrix.hpp"
#include "basic_layer.hpp"
#include "arithmetic_layers.hpp"

namespace ml {

template<typename Field>
class Layer;

template<typename Field>
class Tensor: public Matrix<Field>, protected std::enable_shared_from_this<Tensor<Field>> {
template<typename FField>
friend std::ostream& operator<<(std::ostream& out, const Tensor<FField>& m);

template<typename FField>
friend std::istream& operator>>(std::istream& out, Tensor<FField>& m);

template<typename FField>
friend class Layer;

private:
    std::shared_ptr<Layer<Field>> from_;
    Matrix<Field> grad;
    using Matrix<Field>::matrix;
    using Matrix<Field>::m_;
    using Matrix<Field>::n_;
    bool requires_grad=true;
    
public:
    using Matrix<Field>::operator[];
    using Matrix<Field>::operator+=;
    using Matrix<Field>::operator-=;
    using Matrix<Field>::operator*=;
    using Matrix<Field>::sum;
    using Matrix<Field>::size;
    using Matrix<Field>::transpose;

    Tensor() = default;
    Tensor(size_t m, size_t n, std::shared_ptr<Layer<Field>> from=nullptr): 
        Matrix<Field>(m, n), 
        from_(std::move(from)),
        grad(m, n, 0),
        requires_grad(requires_grad) {}

    Tensor(std::pair<size_t, size_t> size, std::shared_ptr<Layer<Field>> from=nullptr): 
        Tensor(size.first, size.second, std::move(from), requires_grad) {}

    Tensor(const Matrix<Field>& other, std::shared_ptr<Layer<Field>> ptr=nullptr): 
        Matrix<Field>(other), 
        from_(ptr), 
        grad(m_, n_, 0),
        requires_grad(requires_grad){}

    Tensor(Matrix<Field>&& other, std::shared_ptr<Layer<Field>> ptr=nullptr):
        Matrix<Field>(std::move(other)), 
        from_(std::move(ptr)), 
        grad(m_, n_, 0),
        requires_grad(requires_grad) {}

    /*
        Warning! If one want to move/assign/copy Tensor then 
        grad of vector won't be updated it the Tensor
        already has had layer.
        Otherwise implementation become harder. And it is also
        useless to support this logic.
    */

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) = default;

    void backward() {
        backward(Matrix<Field>(m_, n_));
    }

    void backward(const Matrix<Field>& grad) {
        this->grad += grad;
        if (from_) {
            from_->backward_(grad);
        }
    }

    void zero_grad() {
        grad.make_zero();
        if (from_) {
            from_->zero_grad_();
        }
    }

    void break_graph() {
        if (from_) {
            from_->break_graph_();
        }
        from_.reset();
    }

    Field value() const {
        if (m_ != 1 || n_ != 1) {
            throw BadShape("m > 1 or n > 1. Tensor value");
        }
        return matrix[0][0];
    }

    void detach() {
        requires_grad = false;
    }

    // TODO: add optimizer
    void make_step(Field step) {
        if (requires_grad) {
            *this -= (grad * step);
        }
        if (from_) {
            from_->make_step_(step);
        }
    }
};

template<typename Field>
Tensor<Field> operator*(Tensor<Field>& left, Tensor<Field>& right) {
    auto layer = std::make_shared<Multiplier<Field>>();
    return layer->forward(left, right);
}

template<typename Field>
Tensor<Field> operator+(Tensor<Field>& left, Tensor<Field>& right) {
    auto layer = std::make_shared<Adder<Field>>();
    return layer->forward(left, right);
}

template<typename Field>
Tensor<Field> operator-(Tensor<Field>& left, Tensor<Field>& right) {
    auto layer = std::make_shared<Subtractor<Field>>();
    return layer->forward(left, right);
}

template<typename Field>
Tensor<Field> transpose(Tensor<Field>& tensor) {
    auto layer = std::make_shared<Transposer<Field>>();
    return layer->forward(tensor);
}

template<typename Field>
Tensor<Field> operator==(const Tensor<Field>& left, const Tensor<Field>& right){    
    return static_cast<Matrix<Field>&>(left) == static_cast<Matrix<Field>&>(right);
}

template<typename Field>
Tensor<Field> operator!=(const Tensor<Field>& left, const Tensor<Field>& right){    
    return static_cast<Matrix<Field>&>(left) != static_cast<Matrix<Field>&>(right);
}

template<typename Field>
std::ostream& operator<<(std::ostream& out, const Tensor<Field>& m){
    for(auto& row: m.matrix){
        for(auto& elem: row){
            out << elem << " ";
        }
        out << "\n";
    }
    return out;
}


template<typename Field>
std::istream& operator>>(std::istream& in, Tensor<Field>& m){
    for(auto& row: m.matrix){
        for(auto& elem: row){
            in >> elem;
        }
    }
    return in;
}

} // end of namespace ml

