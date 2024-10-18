//
// DD1388 - Lab 4: Losing Chess
//

#include "ChessPiece.h"
#include "ChessBoard.h"

int ChessPiece::validMove(int to_x, int to_y)
{
}

char32_t ChessPiece::utfRepresentation()
{
    // Implementation goes here
}

char ChessPiece::latin1Representation()
{
    // Implementation goes here
}

ChessPiece::ChessPiece(int x, int y, bool is_white, ChessBoard *board, char sign)
{
    m_x = x;
    m_y = y;
    m_board = board;
    m_sign = sign;
    m_is_white = is_white;
}


bool ChessPiece::capturingMove(int to_x, int to_y)
{
    int result = validMove(to_x, to_y);
    if (result == 2)
        return true;
    return false;
}

bool ChessPiece::nonCapturingMove(int to_x, int to_y)
{
    int result = validMove(to_x, to_y);
    if (result == 1)
        return true;
    return false;
}

vector<ChessMove> ChessPiece::capturingMoves()
{
    cout << "ChessPiece capturing moves" << endl;
}

vector<ChessMove> ChessPiece::nonCapturingMoves()
{
    cout << "ChessPiece non capturing moves" << endl;
}

int ChessPiece::valid_return(int to_x, int to_y)
{
    if(m_board->getPiece(to_x, to_y) != nullptr)
    {
        if (m_board->getPiece(to_x, to_y)->m_is_white == m_is_white ||
        !m_board->getPiece(to_x, to_y)->m_is_white == !m_is_white)
            return 0;
        return 2;
    }
  return 1;
}

char ChessPiece::get_sign()
{
    return m_sign;
}

int ChessPiece::x()
{
    return m_x;
}

int ChessPiece::y()
{
    return m_y;
}

bool ChessPiece::is_white()
{
    return m_is_white;
}

