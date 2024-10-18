bintree:
	g++ --std=c++17 -c bintree.cpp
all:
	python3 ~/cxxtest-4.4/bin/cxxtestgen --error-printer -o myrunner.cpp mytest.h
	g++ -std=c++17 -c -o bintree.o bintree.cpp
	g++ -o mytest_runner myrunner.cpp -I ~/cxxtest-4.4 bintree.o
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=testsoutput.txt ./mytest_runner
	g++ -std=c++17 -o testbalance bintree.cpp testbalance.cpp 
	./testbalance
tests:
	./mytest_runner
	./testbalance
balance:
	g++ --std=c++17 -o bintree bintree.cpp testbalance.cpp
	./bintree
