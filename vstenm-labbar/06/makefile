all:
	g++ --std=c++17 Lambdas.cpp
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=testsoutput.txt ./a.out
mac:
	g++ --std=c++11 Lambdas.cpp
	./a.out
hyenas:
	g++ -std=c++17 -pthread main.cpp
	./a.out
