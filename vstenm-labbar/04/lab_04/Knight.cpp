//
// DD1388 - Lab 4: Losing Chess
//

#include "Knight.h"

// Implement method definitions here

int Knight::validMove(int to_x, int to_y)
{
    if (to_x < 0 || to_x > 7 || to_y < 0 || to_y > 7)
        return 0;

    return valid_return(to_x, to_y);
}

vector<ChessMove> Knight::capturingMoves()
{
    return moves(2);
}

vector<ChessMove> Knight::nonCapturingMoves()
{

    return moves(1);
}

vector<ChessMove> Knight::moves(int val)
{

    // Följande idé är tagen ifrån https://stackoverflow.com/questions/19372622/how-do-i-generate-all-of-a-knights-moves
    vector<tuple<int, int>> firstMoves = cartestianProduct({-1, 1}, {-2, 2});
    vector<tuple<int, int>> secondMoves = cartestianProduct({-2, 2}, {-1, 1});

    firstMoves.insert(firstMoves.end(), secondMoves.begin(), secondMoves.end());
    vector<ChessMove> m_moves;

    for (tuple<int, int> t : firstMoves)
    {
        int to_y = m_y + get<0>(t);
        int to_x = m_x + get<1>(t);
        if (validMove(to_x, to_y) == val)
            m_moves.push_back(ChessMove(m_y, m_x, to_y, to_x, this));
    }
    return m_moves;
}

char32_t Knight::utfRepresentation()
{
    if (m_is_white)
        return U'\u2658';
    return U'\u265E';
}
char Knight::latin1Representation()
{
    if (m_is_white)
        return 'N';
    return 'n';
}

vector<tuple<int, int>> cartestianProduct(vector<int> first, vector<int> second)
{
    vector<tuple<int, int>> result;
    for (int i = 0; i < first.size(); i++)
        for (int j = 0; j < second.size(); j++)
            result.push_back(make_tuple(first[i], second[j]));

    return result;
}