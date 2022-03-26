#pragma once

#include "matrix.hpp"
#include "tensor.hpp"

namespace nn{

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
};

} // end of namespace nn



// void backward(const Matrix<Field>& grad_other){
//     //	I can't create grad_ptr in constructors 
//     //	otherwise there is infinite recursion
//     if(!grad_ptr){
//         grad_ptr = std::make_shared<Matrix<Field>>(grad_other.size(), 0.0);
//     }

//     //	the most important step
//     *grad_ptr += grad_other;

//     if(layer_ptr){	
//         //	I need to do change_res_ptr due to 
//         //	occurs calls of copy constructors and I 
//         //	don't have original matrix
//         layer_ptr->res_ptr = this;

//         //	call backward of the layer. It knows how to count grad
//         layer_ptr->backward();
//     }
// }

// //	I need this one for loss functions
// void backward(){
//     if(!grad_ptr){
//         grad_ptr = std::make_shared<Matrix<Field>>();
//     }
//     if(layer_ptr){
//         layer_ptr->res_ptr = this;
//         layer_ptr->backward();
//     }
// }

// void make_step(double step){
//     //	Here I call layer, because it knows how to change weights
//     if(layer_ptr){
//         layer_ptr->make_step(step);
//     }
// }

// void break_graph(){
//     //	with help of recursion I will break the graph
//     if(layer_ptr){
//         layer_ptr->break_graph();
//         layer_ptr.reset();
//     }
//     //	I can't do it in another order
//     //	because I destory connection between them
// }

// void zero_grad(){
//     //	make own grad equal to zero and push instruction down
//     grad_ptr->make_zero();
//     if(layer_ptr){
//         layer_ptr->zero_grad();
//     }
// }

// const Matrix<Field>& get_grad() const{
//     return *grad_ptr;
// }

// class Layer: public std::enable_shared_from_this<Layer>{
// 	public:
// 		Matrix<double>* res_ptr=nullptr;

// 		//	For pushing gradient deeper
// 		virtual void backward() = 0;

// 		//	I have different methods with different number of input arguments
// 		virtual Matrix<double> forward(Matrix<double>&, Matrix<double>&){
// 			throw std::runtime_error("One tries to call forward(Matrix<double>&, Matrix<double>&)");
// 			return Matrix<double>();
// 		}

// 		virtual Matrix<double> forward(Matrix<double>&){
// 			throw std::runtime_error("One tries to call forward(Matrix<double>&)");
// 			return Matrix<double>();
// 		}

// 		//	For change weights
// 		virtual void make_step(double) = 0;
// 		virtual void zero_grad() = 0;

// 		//	For breaking graph because on every iteration it has to been builded again
// 		virtual void break_graph() = 0;
// 		virtual ~Layer() = default;
// };
