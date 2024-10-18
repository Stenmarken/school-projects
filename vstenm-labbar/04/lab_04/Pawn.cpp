//
// DD1388 - Lab 4: Losing Chess
//

#include "Pawn.h"

// Implement method definitions here

int Pawn::validMove(int to_x, int to_y)
{
    if (to_x < 0 || to_x > 7 || to_y < 0 || to_y > 7)
        return 0;

    return valid_return(to_x, to_y);
}

vector<ChessMove> Pawn::capturingMoves()
{
    vector<ChessMove> moves;
    if (m_is_white)
    {
        if (validMove(m_x - 1, m_y - 1) == 2)
            moves.push_back(ChessMove(m_y, m_x, m_y - 1, m_x - 1, this));
        if (validMove(m_x - 1, m_y + 1) == 2)
            moves.push_back(ChessMove(m_y, m_x, m_y + 1, m_x - 1, this));
    }
    else
    {
        if (validMove(m_x + 1, m_y - 1) == 2)
            moves.push_back(ChessMove(m_y, m_x, m_y - 1, m_x + 1, this));
        if (validMove(m_x + 1, m_y + 1) == 2)
            moves.push_back(ChessMove(m_y, m_x, m_y + 1, m_x + 1, this));
    }
    return moves;
}

vector<ChessMove> Pawn::nonCapturingMoves()
{
    vector<ChessMove> moves;
    if (m_is_white && validMove(m_x - 1, m_y) == 1)
    {
        if (validMove(m_x - 1, m_y) == 1)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y, m_x - 1, this));
            if (m_x == 6 && validMove(m_x - 2, m_y) == 1)
                moves.push_back(ChessMove(m_y, m_x, m_y, m_x - 2, this));
        }
    }
    else if (!m_is_white && validMove(m_x + 1, m_y) == 1)
    {
        moves.push_back(ChessMove(m_y, m_x, m_y, m_x + 1, this));
        if (m_x == 1 && validMove(m_x + 2, m_y) == 1)
            moves.push_back(ChessMove(m_y, m_x, m_y, m_x + 2, this));
    }

    return moves;
}

char32_t Pawn::utfRepresentation()
{
    if (m_is_white)
        return U'\u2659';
    return U'\u265F';
}

char Pawn::latin1Representation()
{
    if (m_is_white)
        return 'P';
    return 'p';
}
