default:
	g++ -std=c++11 -c Matrix.cpp

	python3 ~/cxxtest-4.3/bin/cxxtestgen --error-printer -o myrunner.cpp mytest.h
	g++  -std=c++11 -c -o Matrix.o Matrix.cpp
	g++  -std=c++11 -o mytest_runner myrunner.cpp -I $$HOME/cxxtest-4.3 Matrix.o
	./mytest_runner
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./mytest_runner