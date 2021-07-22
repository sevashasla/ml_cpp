#include <iostream>
#include <vector>
#include <list>
#include <memory>

using std::cin, std::cout, std::vector;


struct A{
	struct B{
		A* ptr;

		void f(){
			ptr->f();
		}
	};

	B& f(int){
		B* b = new B;
		return *b;
	}

	void f(){
		cout << 1;
	}
};


int main(){
	

	return 0;
}

