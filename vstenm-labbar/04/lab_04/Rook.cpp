//
// DD1388 - Lab 4: Losing Chess
//

#include "Rook.h"

// Implement method definitions here

vector<ChessMove> Rook::capturingMoves()
{
    int val = 2;
    vector<ChessMove> moves;

    bool upwards = true;
    bool downwards = true;
    bool leftwards = true;
    bool rightwards = true;

    for (int i = 1; i < 8; i++)
    {
        if (validMove(m_x - i, m_y) == 0)
            upwards = false;
        
        if (validMove(m_x + i, m_y) == 0)
            downwards = false;
       
        if (validMove(m_x, m_y - i) == 0)
            leftwards = false;
        
        if (validMove(m_x, m_y + i) == 0)
            rightwards = false;

        if (validMove(m_x - i, m_y) == val && upwards)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y, m_x - i, this));
            upwards = false;
        }
        if (validMove(m_x + i, m_y) == val && downwards)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y, m_x + i, this));
            downwards = false;
        }
        if (validMove(m_x, m_y - i) == val && leftwards)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y - i, m_x, this));
            leftwards = false;
        }
        if (validMove(m_x, m_y + i) == val && rightwards)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y + i, m_x, this));
            rightwards = false;
        }
    }
    return moves;
}

vector<ChessMove> Rook::nonCapturingMoves()
{
    int val = 1;
    vector<ChessMove> moves;

    bool upwards = true;
    bool downwards = true;
    bool leftwards = true;
    bool rightwards = true;

    for (int i = 1; i < 8; i++)
    {
        int down_val = validMove(m_x + i, m_y);
        if (down_val == 0 || down_val == 2)
            downwards = false;

        int up_val = validMove(m_x - i, m_y);
        if (up_val == 0 || up_val == 2)
            upwards = false;

        int left_val = validMove(m_x, m_y - i);
        if (left_val == 0 || left_val == 2)
            leftwards = false;

        int right_val = validMove(m_x, m_y + i);
        if (right_val == 0 || right_val == 2)
            rightwards = false;

        if (upwards)
            moves.push_back(ChessMove(m_y, m_x, m_y, m_x - i, this));
        if (downwards)
            moves.push_back(ChessMove(m_y, m_x, m_y, m_x + i, this));
        if (leftwards)
            moves.push_back(ChessMove(m_y, m_x, m_y - i, m_x, this));
        if (rightwards)
            moves.push_back(ChessMove(m_y, m_x, m_y + i, m_x, this));
    }
    return moves;
}

int Rook::validMove(int to_x, int to_y)
{
    if (to_x < 0 || to_x > 7 || to_y < 0 || to_y > 7)
        return 0;

    return valid_return(to_x, to_y);
}

char32_t Rook::utfRepresentation()
{
    if (m_is_white)
        return U'\u2656';
    return U'\u265C';
}
char Rook::latin1Representation()
{
    if (m_is_white)
        return 'R';
    return 'r';
};

bool Rook::m_is_white_func()
{
    return m_is_white;
}
