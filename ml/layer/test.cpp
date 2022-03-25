#include <iostream>
// #include <memory>
#include "tensor.hpp"
// #include <layer/matrix.hpp>


using namespace std;

struct A{
    void print() {
        cout << "A\n";
    }
};

struct B: A{
    using A::print;
};


int main() {
    nn::Tensor<int> a(5, 3);
    cout << a << "\n";
}