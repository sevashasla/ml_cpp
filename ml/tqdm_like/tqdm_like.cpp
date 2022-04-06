#include "tqdm_like.hpp"

namespace tqdm_like {


ProgressBar::ProgressBar(size_t num_iters, size_t num_stars=100): num_iters_(num_iters), num_stars_(num_stars) {
	std::cout << std::string(num_stars_, '-');
	std::cout << "    ";
	std::cout << "0 /" << num_iters_;
}

size_t ProgressBar::getIter(size_t start, size_t i, size_t step) {
	return (i - start) / step;
}

void ProgressBar::update(size_t iter) {
	std::cout << "\r";

	size_t count = iter * num_stars_ / num_iters_;
	std::cout << std::string(count, '*');
	std::cout << std::string(num_stars_ - count, '-');
	std::cout << "    ";
	std::cout << iter << "/" << num_iters_;
}


void tqdm_like(
	size_t start, 
	size_t end, 
	size_t step, 
	std::function<void()> task
){
	ProgressBar pb(ProgressBar::getIter(start, end, step));
	
	for(size_t i = start; i < end; i += step) {
		pb.update(ProgressBar::getIter(start, i, step) + 1);

		// call function
		task();
	}
	std::cout << "\n";
}

} // end of namespace tdqm_like

