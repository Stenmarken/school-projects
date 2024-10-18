default:
	g++ -c hello.cpp
	g++ -o hello main.cpp hello.o

	python3 $$HOME/cxxtest-4.4/bin/cxxtestgen --error-printer -o myrunner.cpp mytest.h
	g++ -c -o count_if_followed_by.o count_if_followed_by.cpp
	g++ -o mytest_runner myrunner.cpp -I $$HOME/cxxtest-4.4 count_if_followed_by.o
	./mytest_runner