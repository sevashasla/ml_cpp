#include <iostream>
// #include <memory>
#include "tensor.hpp"
// #include "matrix.hpp"

struct A{
    int x = 0;
};

struct B: A{
    int y = 1;
};

using namespace std;

int main() {
    nn::Tensor<double> a(5, 3);
    nn::Tensor<double> b(3, 5);
    auto c = a * b;
    c.backward();
    // c.break_graph();
}
