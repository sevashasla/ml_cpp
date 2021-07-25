#include "ml.h"
// #include "matrix.h"


using std::cin, std::cout;
int main(){
	nn::Sequential<
		nn::Linear<1, 2>
	>
	std::shared_ptr<nn::nnLayer> loss_fn_ptr = std::make_shared<nn::CrossEntropyLoss<2>>();


	std::vector<std::vector<double>> xv({
		
	});
	std::vector<std::vector<double>> yv({
		
	});

	Matrix<double> x(xv);
	Matrix<double> y(yv);

	int num_epoch = 10000;
	while(num_epoch){
		Matrix<double> out = seq.forward(x);
		Matrix<double> loss = loss_fn_ptr->forward(y, out);
		
		loss.backward(Matrix<double>());
		loss.make_step(1e-3);
		loss.zero_grad();
		loss.break_graph();

		if(num_epoch % 50 == 0){
			cout << loss;
		}
		--num_epoch;
	}

	Matrix<double> out = seq.forward(x);
	for(size_t i = 0; i < out.num_rows(); ++i){
		cout << y[i][0] << " " << out[i][0] << "\n";
	}


	return 0;
}

