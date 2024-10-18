//
// DD1388 - Lab 4: Losing Chess
//

#include "Bishop.h"
using namespace std;

// Implement method definitions here

int Bishop::validMove(int to_x, int to_y)
{
    if (to_x < 0 || to_x > 7 || to_y < 0 || to_y > 7)
        return 0;

    return valid_return(to_x, to_y);
}

vector<ChessMove> Bishop::nonCapturingMoves()
{
    vector<ChessMove> moves;
    bool up_right = true;
    bool up_left = true;
    bool down_right = true;
    bool down_left = true;

    for (int i = 1; i < 8; i++)
    {
        int down_right_val = validMove(m_x + i, m_y + i);
        if (down_right_val == 0 || down_right_val == 2)
            down_right = false;

        int up_right_val = validMove(m_x - i, m_y + i);
        if (up_right_val == 0 || up_right_val == 2)
            up_right = false;

        int down_left_val = validMove(m_x + i, m_y - i);
        if (down_left_val == 0 || down_left_val == 2)
            down_left = false;

        int up_left_val = validMove(m_x - i, m_y - i);
        if (up_left_val == 0 || up_left_val == 2)
            up_left = false;

        if (down_right)
            moves.push_back(ChessMove(m_y, m_x, m_y + i, m_x + i, this));       
        if (up_right)
            moves.push_back(ChessMove(m_y, m_x, m_y + i, m_x - i, this));
        if (down_left)
            moves.push_back(ChessMove(m_y, m_x, m_y - i, m_x + i, this));
        if (up_left)
            moves.push_back(ChessMove(m_y, m_x, m_y - i, m_x - i, this));
    }

    return moves;
}

vector<ChessMove> Bishop::capturingMoves()
{
    int val = 2;
    vector<ChessMove> moves;
    bool up_right = true;
    bool up_left = true;
    bool down_right = true;
    bool down_left = true;

    for (int i = 1; i < 8; i++)
    {
        if (validMove(m_x + i, m_y + i) == 0)
            down_right = false;

        if (validMove(m_x - i, m_y + i) == 0)
            up_right = false;

        if (validMove(m_x + i, m_y - i) == 0)
            down_left = false;

        if (validMove(m_x - i, m_y - i) == 0)
            up_left = false;

        if (validMove(m_x + i, m_y + i) == val && down_right)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y + i, m_x + i, this));
            down_right = false;
        }
        if (validMove(m_x - i, m_y + i) == val && up_right)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y + i, m_x - i, this));
            up_right = false;
        }
        if (validMove(m_x + i, m_y - i) == val && down_left)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y - i, m_x + i, this));
            down_left = false;
        }
        if (validMove(m_x - i, m_y - i) == val && up_left)
        {
            moves.push_back(ChessMove(m_y, m_x, m_y - i, m_x - i, this));
            up_left = false;
        }
    }
    return moves;
}

char32_t Bishop::utfRepresentation()
{
    if (m_is_white)
        return U'\u2657';
    return U'\u265D';
}
char Bishop::latin1Representation()
{
    if (m_is_white)
        return 'B';
    return 'b';
}
