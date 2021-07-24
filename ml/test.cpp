#include <iostream>
#include <vector>
#include <list>
#include <memory>

using std::cin, std::cout, std::vector;


struct A: public std::enable_shared_from_this<A>{};

struct B: public A{
	std::shared_ptr<A> get_ptr_B(){
		return shared_from_this();
	}
};

int main(){
	std::shared_ptr<B> b_ptr = std::make_shared<B>();
	b_ptr->get_ptr_B();
	

	return 0;
}

