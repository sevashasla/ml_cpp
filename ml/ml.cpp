#include "matrix.h"
using std::cin, std::cout;

namespace nn{
	class Layer{
		virtual Matrix<double> forward(const Matrix<double>& x) const = 0;
		virtual void backward() = 0;
	};

	template<size_t N>
	class ReLU: Layer{
	private:
		Matrix<double> grad;
	public:
		
		auto forward(const Matrix<double>& x) const{

		}

		void backward(){

		}

	};

	template<typename... Layers>
	class Sequantial{
	private:
	public:
		static const size_t length = sizeof...(Layers);


	};

}



int main(){

	return 0;
}

