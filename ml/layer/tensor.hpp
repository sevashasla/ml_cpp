#pragma once

#include "matrix.hpp"
#include "basic_layer.hpp"

namespace nn{

template<typename Field>
class Layer;

template<typename Field>
class Tensor: Matrix<Field>, std::enable_shared_from_this<Tensor<Field>> {
template<typename FField>
friend std::ostream& operator<<(std::ostream& out, const Tensor<FField>& m);

template<typename FField>
friend std::istream& operator>>(std::istream& out, Tensor<FField>& m);

private:
    std::shared_ptr<Layer<Field>> from_;
    using Matrix<Field>::matrix;
    
public:
    using Matrix<Field>::Matrix;
    using Matrix<Field>::operator[];
    using Matrix<Field>::operator+=;
    using Matrix<Field>::operator-=;
    using Matrix<Field>::operator*=;

    void abobus() {
        from_->backward();
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

}
