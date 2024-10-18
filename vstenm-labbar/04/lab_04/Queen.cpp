//
// DD1388 - Lab 4: Losing Chess
//

#include "Queen.h"

// Implement method definitions here
vector<ChessMove> Queen::capturingMoves()
{
    vector<ChessMove> cm = Rook::capturingMoves();
    vector<ChessMove> b_moves = Bishop::capturingMoves();
    cm.insert(cm.end(), b_moves.begin(), b_moves.end());
    return cm;
}

vector<ChessMove> Queen::nonCapturingMoves()
{
    vector<ChessMove> cm = Rook::nonCapturingMoves();
    vector<ChessMove> b_moves = Bishop::nonCapturingMoves();
    cm.insert(cm.end(), b_moves.begin(), b_moves.end());
    return cm;
}

int Queen::validMove(int to_x, int to_y)
{
    if (to_x < 0 || to_x > 7 || to_y < 0 || to_y > 7)
        return 0;

    return valid_return(to_x, to_y);
}

char32_t Queen::utfRepresentation()
{
if (Rook::m_is_white_func() == true)
        return U'\u2655';
    else
        return U'\u265B';
}

char Queen::latin1Representation()
{
    if (Rook::m_is_white_func() == true)
        return 'Q';
    else
        return 'q';
}
