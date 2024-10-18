build:
	g++ -I ../../02 -std=c++11 ChessBoard.cpp Bishop.cpp ChessPiece.cpp King.cpp Knight.cpp Pawn.cpp Queen.cpp Rook.cpp sample.cpp -o debug
tests:
	g++ -I ../../02 -std=c++11 ChessBoard.cpp Bishop.cpp ChessPiece.cpp King.cpp Knight.cpp Pawn.cpp Queen.cpp Rook.cpp tests.cpp -o tests
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes --verbose --log-file=valgrind-out.txt ./tests < ./tests.in
ai:
	g++ -I ../../02 -std=c++11 ChessBoard.cpp Bishop.cpp ChessPiece.cpp King.cpp Knight.cpp Pawn.cpp Queen.cpp Rook.cpp main.cpp -o ai
debug:
	g++ -I ../../02 -std=c++11 ChessBoard.cpp Bishop.cpp ChessPiece.cpp King.cpp Knight.cpp Pawn.cpp Queen.cpp Rook.cpp sample.cpp -o debug
tests-mac:
	g++ -I ../../02 -std=c++11 ChessBoard.cpp Bishop.cpp ChessPiece.cpp King.cpp Knight.cpp Pawn.cpp Queen.cpp Rook.cpp tests.cpp -o tests
	./tests < ./tests.in
