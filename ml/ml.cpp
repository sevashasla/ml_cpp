#include "ml.h"
// #include "matrix.h"


using std::cin, std::cout;
int main(){
	constexpr size_t num_classes = 2;

	nn::Sequential<
		nn::Linear<1, 2>,
		nn::BatchNorm,
	> seq;


	std::shared_ptr<nn::nnLayer> loss_fn_ptr = std::make_shared<nn::CrossEntropyLoss<num_classes>>();

	std::vector<std::vector<double>> xv({
		{-11.092624349394143}, {-10.4448410684391}, {-9.811151112664817}, {-9.837675434205272}, 
		{-10.668374789942131}, {-1.6173461592329268}, {-10.303916516281474}, {-0.794152276623841}, 
		{-1.1466300855305107}, {-1.8684541393232976}, {-9.257156052556827}, {-9.647166524988995}, 
		{0.08525185826796045}, {-9.682077556411496}, {-0.7587039566841077}, {-1.1710417594110225}, 
		{-11.196980535988288}, {-7.81213709711994}, {-10.372997453743215}, {-0.8205764920740146}, 
		{-10.341566179224177}, {-1.9819771099620271}, {-1.7824501314671677}, {-8.866083116201674}, 
		{0.00024227116135100424}, {-9.875891232661665}, {-2.4139578469451726}, {-2.406718199699357}, 
		{-11.370829823899857}, {-9.151551856068068}, {-11.44182630908269}, {-10.036408012919152}, 
		{-10.220040646263461}, {-1.8319881134989553}, {-2.346732606068119}, {-9.574218149588988}, 
		{-1.359389585992692}, {-1.340520809891421}, {-1.6087521511724905}, {-2.351220657673829}, 
		{-1.5394009534668904}, {-0.19745196890354544}, {-8.876294795417436}, {-8.370061750504195}, 
		{-2.760179083161441}, {-1.8513954583101344}, {-2.3308060367853387}, {-11.855694368099854}, 
		{-2.187731658211975}, {-10.02232945952888}, {-8.798794623751593}, {-9.5942208618623}, 
		{-10.263931009656723}, {-10.617713347601232}, {-8.723956573494325}, {-2.8020781039706595}, 
		{-0.5257904636130821}, {-1.9274479855745354}, {-0.757969185355724}, {-9.799412783526332}, 
		{-9.767617768288718}, {-1.4686444212810534}, {-1.3739725806942609}, {-2.7768702545837973}
	});

	std::vector<std::vector<double>> yv({
		{1}, {1}, {1}, {1}, 
		{1}, {0}, {1}, {0}, 
		{0}, {0}, {1}, {1}, 
		{0}, {1}, {0}, {0}, 
		{1}, {1}, {1}, {0}, 
		{1}, {0}, {0}, {1}, 
		{0}, {1}, {0}, {0}, 
		{1}, {1}, {1}, {1}, 
		{1}, {0}, {0}, {1}, 
		{0}, {0}, {0}, {0}, 
		{0}, {0}, {1}, {1}, 
		{0}, {0}, {0}, {1}, 
		{0}, {1}, {1}, {1}, 
		{1}, {1}, {1}, {0}, 
		{0}, {0}, {0}, {1}, 
		{1}, {0}, {0}, {0},
	});

	Matrix<double> x(xv);
	Matrix<double> y_fresh(yv);
	Matrix<double> y = static_cast<Matrix<double>>(nn::OneHot<num_classes>(y_fresh));


	// std::shared_ptr<nn::BatchNorm> bn_ptr = std::make_shared<nn::BatchNorm>();
	// std::shared_ptr<nn::Linear<1, 2>> lin_ptr = std::make_shared<nn::Linear<1, 2>>();

	// Matrix<double> out1 = lin_ptr->forward(x);
	// Matrix<double> out2 = bn_ptr->forward(out1);
	// Matrix<double> loss = loss_fn_ptr->forward(y, out1);
	// loss.backward();

	// cout << lin_ptr->get_weight();
	// cout << lin_ptr->get_bias();


	int num_epoch = 1000;
	while(num_epoch){

		Matrix<double> out = seq.forward(x);
		// Matrix<double> out = lin_ptr->forward(x);
		Matrix<double> loss = loss_fn_ptr->forward(y, out);
		loss.backward();

		loss.make_step(1e-3);		
		loss.zero_grad();
		loss.break_graph();


		// if(num_epoch % 50 == 0){
		// 	cout << loss;
		// }
		--num_epoch;
	}

	Matrix<double> out = seq.forward(x);
	Matrix<double> pred_classes = static_cast<Matrix<double>>(nn::predict<num_classes>(out));

	cout << nn::accuracy(y_fresh, pred_classes);
	return 0;
}

