default:
	g++ -c Matrix.cpp

	python3 $$HOME/cxxtest-4.4/bin/cxxtestgen --error-printer -o myrunner.cpp mytest.h
	g++ -c -o Matrix.o Matrix.cpp
	g++ -o mytest_runner myrunner.cpp -I $$HOME/cxxtest-4.4 Matrix.o
	./mytest_runner
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./mytest_runner