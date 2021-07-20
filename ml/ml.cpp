#include "matrix.h"
#include <list>
using std::cin, std::cout;


namespace nn{
	class Layer{
	public:
		virtual Matrix<double> forward(const Matrix<double>&) = 0;
		virtual Matrix<double> backward(const Matrix<double>&) = 0;
		virtual ~Layer(){}
	};

	
	template<size_t N>
	class ReLU: Layer{
	private:
		Matrix<double> grad;
	public:
		
		auto forward(const Matrix<double>& x) {

		}

		void backward(){

		}

	};

	
	class MSELoss: Layer{

	private:
		Matrix<double> grad;
	public:

	};


	template<size_t In, size_t Out>
	class Linear: public Layer{
	/*
		x: N x (In + 1)
		w: (In + 1) x Out
		

		let after it will be f(y) and we have grad_y where y = x*w
		so we need to find dy/dw

		let's look at only one vector => 
		x: 1 x M = (x_1, ... , x_M)
		y = xw: 1 x C = (y_1, ... , y_C)
		grad_y: 1 x C = (df(y)/dy_1, ... , df(y)/dy_C)

			(w_{1 1}, ..., w_{1 C})
		w = 			...			=> y = (x_1 * w_{1 1} + ... + x_M * w_{M 1}, ... , x_1 * w_{1 C} + ... + x_M * w_{M C})
			(w_{M 1}, ..., w_{M C})

		df(y)/dw_{i j} = df(y)/dy_j * dy_j/dw_{i j} = df(y)/dy_j * x_i

					(df(y)/dy_1 * x_1 , ... , df(y)/dy_C * x_1)		(x_1)
		grad_w = 						...						= 	...	 * 	(df(y)/dy_1, ... , df(y)/dy_C) = x^T * grad_y
					(df(y)/dy_1 * x_M , ... , df(y)/dy_C * x_M)		(x_M)





	*/
	private:
		Matrix<double> result;
		Matrix<double> grad;
		Matrix<double> w;

	public:
		Linear(): w(Matrix<double>::random(In + 1, Out)){}

		Matrix<double> forward(const Matrix<double>& x) override{
			if(x.num_columns() != In + 1){
				throw BadShape("x in forward of linear has wrong shape");
			}
			result = x * w;
			return result;
		}

		
		Matrix<double> backward(const Matrix<double>& other_grad) override{
			grad = result.transpose() * other_grad;
			grad /= result.num_rows();
			return grad;
		}

		void make_step(double step){
			grad *= step;
			w -= grad;
		}

		~Linear() = default;
	};

	template<typename... Layers>
	class Sequantial{
	private:
		std::list<Layer*> seq;

		auto add_ones(const Matrix<double>& x_fresh) const{
			Matrix<double> res(x_fresh);
			for(size_t i = 0; i < res.num_rows(); ++i){
				res[i].push_back(1.0);
			}
		}
		template<typename... LLayers>
		void PushBackMany(LLayers&&... layers){}

		template<typename Head, typename... LLayers>
		void PushBackMany(Head&& head, LLayers&&... layers){
			seq.push_back(new Head(std::forward<Head>(head)));
			PushBackMany(std::forward<LLayers>(layers)...);
		}

	public:
		static const size_t length = sizeof...(Layers);
		Sequantial(): Sequantial(Layers()...){}

		template<typename... LLayers>
		Sequantial(LLayers&&... layers){
			PushBackMany(std::forward<LLayers>(layers)...);
		}

		auto forward(const Matrix<double>& x_fresh) {
			auto x = x_fresh;
			for(auto& layer_ptr: seq){
				x = layer_ptr->forward(x);
			}
			return x;
		}

		void backward(){

		}
	};

}



int main(){
	nn::Sequantial<nn::Linear<1, 2>> net;
	try{
		net.forward(Matrix<double>(5, 1));
	} catch(BadShape& x){

	}

	return 0;
}

