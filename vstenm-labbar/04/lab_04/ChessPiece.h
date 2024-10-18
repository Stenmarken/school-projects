//
// DD1388 - Lab 4: Losing Chess
//

#ifndef CHESSPIECE_H
#define CHESSPIECE_H

#include <vector>
#include "ChessMove.h"
#include "ChessBoard.h"

using namespace std;

class ChessPiece {
    friend void ChessBoard::movePiece(ChessMove p);
protected:                               // protected will cause problems
    ChessBoard* m_board;
    char m_sign;
    int m_x, m_y;
    bool m_is_white;
    /**
     * Returns 0 if target square is unreachable.
     * Returns 1 if target square is reachable and empty.
     * Returns 2 if move captures a piece.
     */
    virtual int validMove(int to_x, int to_y);

public:
    // Constructor
    ChessPiece(int x, int y, bool is_white, ChessBoard * board, char sign);
    /**
     * Checks if this move is valid for this piece and captures
     * a piece of the opposite color.
     */
    bool capturingMove(int to_x, int to_y);
    /**
     * Checks if this move is valid but does not capture a piece.
     */
    bool nonCapturingMove(int to_x, int to_y);
    virtual vector<ChessMove> capturingMoves();
    virtual vector<ChessMove> nonCapturingMoves();

    /**
    * For testing multiple inheritance
    */
    int unnecessary_int;

    int valid_return(int to_x, int to_y);
    char get_sign();

    virtual char32_t utfRepresentation();     // may be implemented as string
    virtual char latin1Representation();

    int x();
    int y();
    bool is_white();
};


#endif //CHESSPIECE_H
