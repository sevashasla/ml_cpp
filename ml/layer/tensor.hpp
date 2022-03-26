#pragma once

#include "matrix.hpp"
#include "basic_layer.hpp"
#include "arithmetic_layers.hpp"

namespace nn{

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
    
public:
    using Matrix<Field>::operator[];
    using Matrix<Field>::operator+=;
    using Matrix<Field>::operator-=;
    using Matrix<Field>::operator*=;
    using Matrix<Field>::size;
    Tensor() = default;
    Tensor(size_t m, size_t n): Matrix<Field>(m, n){}
    Tensor(const Matrix<Field>& other, std::shared_ptr<Layer<Field>> ptr): Matrix<Field>(other), from_(ptr){}
    Tensor(Matrix<Field>&& other, std::shared_ptr<Layer<Field>> ptr): Matrix<Field>(std::move(other)), from_(std::move(ptr)){}

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
            from_->make_zero_();
        }
    }

    void break_graph() {
        if (from_) {
            from_->break_graph_();
        }
        from_.reset();
    }

    // add optimizer
    void make_step(Field step) {
        *this -= static_cast<Matrix<Field>>(step) * grad;
        if (from_) {
            from_->make_step_(step);
        }
    }
};

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

template<typename Field>
Tensor<Field> operator+(Tensor<Field>& left, Tensor<Field>& right) {
    auto layer = std::make_shared<Adder<Field>>();
    return layer->forward(left, right);
}

} // end of namespace nn

