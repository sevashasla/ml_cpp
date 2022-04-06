#include "tqdm_like.cpp" // BAD BAD BAD

int main() {
	tqdm_like::tqdm_like(0, 100000, 1, [](){});
	return 0;
}
