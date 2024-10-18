//
// DD1388 - Lab 4: Losing Chess
//

#ifndef KING_H
#define KING_H

#include "ChessPiece.h"

class King : public ChessPiece {
    
    // Override virtual methods from ChessPiece here
    public:
        King(int x, int y, bool is_white, ChessBoard * board, char sign) : ChessPiece(x, y, is_white, board, sign) {}
        int validMove(int to_x, int to_y);
        vector<ChessMove> capturingMoves();
        vector<ChessMove> nonCapturingMoves();
        vector<ChessMove> moves(int val);
        virtual char32_t utfRepresentation();     // may be implemented as string
        virtual char latin1Representation();
};


#endif //KING_H
