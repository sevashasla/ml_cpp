#include "ml.h"
// #include "matrix.h"


using std::cin, std::cout;
int main(){
	nn::Sequential<
		nn::Linear<5, 1>,
		nn::Linear<1, 2>
	> seq;
	

	return 0;
}

