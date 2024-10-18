//
// DD1388 - Lab 4: Losing Chess
//

#ifndef ROOK_H
#define ROOK_H

#include "ChessPiece.h"

class Rook : virtual public ChessPiece {
    // Override virtual methods from ChessPiece here
    public:
        Rook(int x, int y, bool is_white, ChessBoard * board, char sign) : ChessPiece(x, y, is_white, board, sign) {}
        vector<ChessMove> capturingMoves();
        vector<ChessMove> nonCapturingMoves();
        int validMove(int to_x, int to_y);
        char32_t utfRepresentation();     // may be implemented as string
        char latin1Representation();
        bool m_is_white_func();
};


#endif //ROOK_H
