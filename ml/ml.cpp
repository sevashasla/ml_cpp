#include "ml.h"
// #include "matrix.h"


using std::cin, std::cout;
int main(){
	std::shared_ptr<nn::Linear<1, 1>> linear_ptr = std::make_shared<nn::Linear<1, 1>>();
	std::shared_ptr<nn::MSELoss> loss_fn_ptr = std::make_shared<nn::MSELoss>();

	std::vector<std::vector<double>> xv({
		{-1.1147214599182884},
		{0.6823555673689515},
		{-0.3756996819598796},
		{0.033522497470759714},
		{-0.891925967669783},
		{1.6435863149287389},
		{1.9093465791727442},
		{-0.8197135388933674},
		{1.0327433842598033},
		{0.882134641872002},
		{1.3303151901075494},
		{-0.06240213820894151},
		{0.13501345885219898},
		{-1.4751306512958913},
		{0.1532815716512277},
		{-1.6548957126744046},
	});
	std::vector<std::vector<double>> yv({
		{-90.6923904806204},
		{53.80272414889312},
		{-32.49303612344725},
		{3.0586067101168006},
		{-70.15888338555139},
		{133.27128868276418},
		{153.31565318436822},
		{-67.12405307539548},
		{81.06405114695735},
		{70.01329837576486},
		{108.49244906172706},
		{-3.264139186889557},
		{11.721245005116101},
		{-117.99237601359168},
		{10.500537196138952},
		{-134.03303791290068}
	});

	Matrix<double> x(xv);
	Matrix<double> y(yv);

	int num_epoch = 10000;
	while(num_epoch){
		Matrix<double> out = linear_ptr->forward(x);
		Matrix<double> loss = loss_fn_ptr->forward(y, out);
		
		loss.backward(Matrix<double>());
		loss.make_step(1e-3);
		loss.zero_grad();
		loss.break_graph();

		if(num_epoch % 50 == 0){
			cout << loss << "\n";
		}
		--num_epoch;
	}

	Matrix<double> out = linear_ptr->forward(x);
	for(size_t i = 0; i < out.num_rows(); ++i){
		cout << y[i][0] << " " << out[i][0] << "\n";
	}
	

	return 0;
}

