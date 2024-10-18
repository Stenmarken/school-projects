all: Complex.cpp Complex.h mytest.h
	g++ -c Complex.cpp
	python3 $$HOME/cxxtest-4.4/bin/cxxtestgen --error-printer -o myrunner.cpp mytest.h
	g++ -c -o Complex.o Complex.cpp
	g++ -o mytest_runner myrunner.cpp -I $$HOME/cxxtest-4.4 Complex.o

tests:
	./mytest_runner
