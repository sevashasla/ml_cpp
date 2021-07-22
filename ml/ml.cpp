// #include "ml.h"
#include "matrix.h"

using std::cin, std::cout;
int main(){
	Matrix<double> x(1, 1, 1.0);
	Matrix<double> y(1, 1, 1.0);
	Matrix<double> z = x + y;

	Matrix<double> ggg(1, 1, 1.0);
	z.backward(ggg);

	return 0;
}

