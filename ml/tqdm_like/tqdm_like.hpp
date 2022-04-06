#pragma once

#include <iostream>
#include <functional>
#include <string>
#include <chrono>

namespace tqdm_like {

class ProgressBar{
private:
	size_t num_iters_;
	size_t num_stars_;
	decltype(std::chrono::steady_clock::now()) time_start_;

private:
	size_t getStarsCount(size_t /*iter*/);

	double waitTime(size_t /*iter*/);

public:
	ProgressBar(size_t /*num_iters*/, size_t /*num_stars=100*/);

	static size_t getIter(size_t /*start*/, size_t /*i*/, size_t /*step*/);


	void update(size_t /*iter*/);
};

void tqdm_like(
	size_t /*start*/, 
	size_t /*end*/, 
	size_t /*step*/, 
	std::function<void()> /*task*/
);

} // end of namespace tqdm_like
