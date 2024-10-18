//
// DD1388 - Lab 4: Losing Chess
//

#ifndef KNIGHT_H
#define KNIGHT_H

#include "ChessPiece.h"

class Knight : public ChessPiece {
    // Override virtual methods from ChessPiece here
    public:
        Knight(int x, int y, bool is_white, ChessBoard * board, char sign) : ChessPiece(x, y, is_white, board, sign) {}
        int validMove(int to_x, int to_y);
        vector<ChessMove> capturingMoves();
        vector<ChessMove> nonCapturingMoves();
        virtual char32_t utfRepresentation();     // may be implemented as string
        virtual char latin1Representation();
        vector<ChessMove> moves(int val);
};

vector<tuple<int, int>> cartestianProduct(vector<int> first, vector<int> second);


#endif //KNIGHT_H
