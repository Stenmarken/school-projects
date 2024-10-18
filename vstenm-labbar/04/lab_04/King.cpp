//
// DD1388 - Lab 4: Losing Chess
//

#include "King.h"

// Implement method definitions here

int King::validMove(int to_x, int to_y)
{
    if (to_x < 0 || to_x > 7 || to_y < 0 || to_y > 7)
        return 0;
        
    if (abs(to_x - m_x) <= 1 && abs(to_y - m_y) <= 1 &&
        (to_x != m_x || to_y != m_y))
    {
        return valid_return(to_x, to_y);
    }
    return 0;
}

vector<ChessMove> King::capturingMoves()
{
    return moves(2);
}

vector<ChessMove> King::nonCapturingMoves()
{
    return moves(1);
}

vector<ChessMove> King::moves(int val)
{
    vector<tuple<int, int>> changes = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
    vector<ChessMove> cm;
    for(tuple<int, int> t : changes)
    {
        int x = get<0>(t);
        int y = get<1>(t);
        if (validMove(m_x + x, m_y + y) == val)
        {
            ChessMove c = {m_y, m_x, m_y + y, m_x + x, this};
            cm.push_back(c);
        }
    }
    return cm;
}

char32_t King::utfRepresentation()
{
    if (m_is_white)
        return U'\u2654';
    return U'\u265A';
}
char King::latin1Representation()
{
    if (m_is_white)
        return 'K';
    return 'k';
}