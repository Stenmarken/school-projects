//
// DD1388 - Lab 4: Losing Chess
//

#ifndef PAWN_H
#define PAWN_H

#include "ChessPiece.h"

class Pawn : public ChessPiece {
    // Override virtual methods from ChessPiece here
    public:
        Pawn(int x, int y, bool is_white, ChessBoard * board, char sign) : ChessPiece(x, y, is_white, board, sign) {}
        int validMove(int to_x, int to_y);
        vector<ChessMove> capturingMoves();
        vector<ChessMove> nonCapturingMoves();
        virtual char32_t utfRepresentation();     // may be implemented as string
        virtual char latin1Representation();
};


#endif //PAWN_H
