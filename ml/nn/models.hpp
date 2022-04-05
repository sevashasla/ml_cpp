#pragma once

#include "../tensor/tensor.hpp"
#include "../tensor/basic_layer.hpp"
#include <list>

namespace ml::nn::models {
	
	template<typename Field, typename TLayer>
	class SingleLayer{
	private:
		std::shared_ptr<TLayer> layer_;
	public:
		SingleLayer(): layer_(std::make_shared<TLayer>()){}

		// non copyable
		SingleLayer(const SingleLayer&) = delete;
		SingleLayer& operator=(const SingleLayer&) = delete;

		// non movable
		SingleLayer& operator=(SingleLayer&&) = delete;
		SingleLayer(SingleLayer&&) = delete;

		~SingleLayer() = default;

		std::shared_ptr<TLayer> getLayer() const {
			return layer_;
		}

		Tensor<Field> forward(Tensor<Field>& tensor) {
			return layer_->forward(tensor);
			
		}

		Tensor<Field> forward(Tensor<Field>& left, Tensor<Field>& right) {
			return layer_->forward(left, right);
		}
	};


	template<typename Field, typename... Layers>
	class Sequential{
	private:
		std::list<std::shared_ptr<Layer<Field>>> seq;
		std::list<Tensor<Field>> outputs;

		template<typename... LLayers>
		void Add(LLayers&&... layers){}

		template<typename Head, typename... LLayers>
		void Add(){
			seq.push_back(std::make_shared<Head>());
			Add<LLayers...>();
		}

	public:
		static const size_t length = sizeof...(Layers);

		Sequential(){
			Add<Layers...>();
		}

		Tensor<Field> forward(Tensor<Field>& input){
			outputs.clear();
			outputs.push_back(input);
			for(auto& layer_ptr: seq){
				auto& input_current = outputs.back();
				auto output_current = layer_ptr->forward(input_current);
				outputs.push_back(std::move(output_current));
			}
			return outputs.back();
		}

		void backward(const Matrix<Field>& grad){
			outputs.back().backward(grad);
		}

		void make_step(Field step){
			outputs.back().make_step_(step);
		}

		void zero_grad(){
			outputs.back().zero_grad_();
		}

		// Do I really need this?
		void break_graph(){
			outputs.back().break_graph_();
		}

		~Sequential(){}
	};
}

