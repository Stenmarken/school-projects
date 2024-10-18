all:
	g++ -I ../05 -std=c++20 -o node Node.cpp
	./node
test-iterator:
	g++ -c -I ../05 Node.cpp
	python3 $$HOME/cxxtest-4.4/bin/cxxtestgen --error-printer -o myrunner.cpp iterator_tests.h
	g++ -c -o Node.o Node.cpp
	g++ -o mytest_runner myrunner.cpp -I $$HOME/cxxtest-4.4 Node.o
	./mytest_runner
friend:
	g++ -o ./friend/friend_node ./friend/Friend_node.cpp
	./friend/friend_node
