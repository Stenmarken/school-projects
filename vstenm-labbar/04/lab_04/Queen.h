//
// DD1388 - Lab 4: Losing Chess
//

#ifndef QUEEN_H
#define QUEEN_H

#include "ChessPiece.h"
#include "Rook.h"
#include "Bishop.h"

class Queen : virtual public Rook, virtual public Bishop {
    // Override virtual methods from ChessPiece here
    public:
        Queen(int x, int y, bool is_white, ChessBoard * board, char sign) : Bishop(x, y, is_white, board, sign), Rook(x, y, is_white, board, sign), ChessPiece(x, y, is_white, board, sign) {}
        vector<ChessMove> capturingMoves();
        vector<ChessMove> nonCapturingMoves();
        char32_t utfRepresentation();     // may be implemented as string
        char latin1Representation();
        int validMove(int to_x, int to_y);
};


#endif //QUEEN_H
