#include <iostream>
#include <vector>
#include <list>
#include <memory>


using std::cin, std::cout, std::vector;

template<typename T>
void wt() = delete;

// struct Strange{
// 	std::list<int> lt;

// 	template<typename... Args>
// 	Strange(Args&&... args){
// 		lt.insert(args...);
// 	}

// };

struct A{
	virtual void f(){
		cout << 1;
	}
};

struct B: public A{
	void f() override{
		cout << 2;
	}
};

int main(){
	

	return 0;
}

