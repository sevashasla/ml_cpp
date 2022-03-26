#include <iostream>
// #include <memory>
#include "tensor.hpp"
// #include <layer/matrix.hpp>


using namespace std;

int main() {
    nn::Tensor<double> a(5, 3);
    nn::Tensor<double> b(5, 3);
    auto c = a + b;
    c.backward();

}
