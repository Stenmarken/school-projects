//
// DD1388 - Lab 4: Losing Chess
//

#ifndef CHESSMOVE_H
#define CHESSMOVE_H

using namespace std;

class ChessPiece;

struct ChessMove {
    int from_x;
    int from_y;
    int to_x;
    int to_y;

    ChessPiece * piece;   // you can change the position of the chess piece with this pointer.
    ChessMove(int from_x, int from_y, int to_x, int to_y, ChessPiece * piece)
        : from_x(from_x), from_y(from_y), to_x(to_x), to_y(to_y), piece(piece) {}
    ChessMove() {}
};

#endif //CHESSMOVE_H
