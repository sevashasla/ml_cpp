#include "tqdm_like.hpp"

namespace tqdm_like {

////////////////////////////////////////////////////////////

ProgressBar::ProgressBar(size_t num_iters, size_t num_stars=100): 
	num_iters_(num_iters), 
	num_stars_(num_stars),
	time_start_(std::chrono::steady_clock::now()) {
	std::cout << std::string(num_stars_, '-');
	std::cout << "    ";
	std::cout << "0/" << num_iters_;
}

size_t ProgressBar::getIter(size_t start, size_t i, size_t step) {
	return (i - start) / step;
}

size_t ProgressBar::getStarsCount(size_t iter) {
	return iter * num_stars_ / num_iters_;
}

double ProgressBar::waitTime(size_t iter) {
	auto diff = std::chrono::duration<double>(std::chrono::steady_clock::now() - time_start_);
	auto diff_seconds = std::chrono::duration_cast<std::chrono::seconds>(diff);
	return (diff_seconds * (num_iters_ - iter) / iter).count();
}

void ProgressBar::update(size_t iter) {
	size_t count = getStarsCount(iter);
	if (iter == 0 || getStarsCount(iter - 1) == count) {
		return;
	}

	std::cout << "\r";

	std::cout << std::string(count, '*');
	std::cout << std::string(num_stars_ - count, '-');
	std::cout << "    ";
	std::cout << iter << "/" << num_iters_;
	std::cout << ", wait: ~" << waitTime(iter) << "s";
}


////////////////////////////////////////////////////////////

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

