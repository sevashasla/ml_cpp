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

	template<size_t In, size_t Out>
	class Linear: Layer{
	private:
		Matrix<double> grad(In, Out, 0.0);
		Matrix<double> w(In, Out);

	public:
		auto forward(const Matrix<double>& x) const{
			return w * x;
		}

		//get grad?
		void backward(){

		}
	}

	template<typename... Layers>
	class Sequantial{
	private:
		std::list<Layer> seq;

		template<Head, LLayers>
		void PushBackMany(Head&& head, LLayers&&... layers){
			seq.push_back(std::forward<Head>(head));
			PushBackMany(std::forward<LLayers>(layers)...);
		}

	public:
		static const size_t length = sizeof...(Layers);
		Sequantial(): Sequantial(Layers()...){}

		template<LLayers>
		Sequantial(LLayers&&... layers){
			PushBackMany(std::forward<LLayers>(layers)...);
		}




	};

}



int main(){

	return 0;
}

