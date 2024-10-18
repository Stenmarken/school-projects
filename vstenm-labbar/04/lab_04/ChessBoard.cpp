//
// DD1388 - Lab 4: Losing Chess
//

#include "ChessBoard.h"
#include "King.h"
#include "Queen.h"
#include "Rook.h"
#include "Bishop.h"
#include "Knight.h"
#include "Pawn.h"

ChessBoard::ChessBoard(int n)
{
    m_state = Matrix<shared_ptr<ChessPiece>>(n, n);
    row_length = n;
}

ChessBoard::ChessBoard()
{
    const int n = 8;
    m_state = Matrix<shared_ptr<ChessPiece>>(n, n);
    row_length = 8;
}

void ChessBoard::add_chesspiece(shared_ptr<ChessPiece> piece)
{
    if (piece->is_white() == true)
    {
        m_white_pieces.push_back(&(*piece));
    }
    else
    {
        m_black_pieces.push_back(&(*piece));
    }
    m_state(piece->x(), piece->y()) = piece;
}

void ChessBoard::insert_piece(int i, int j, char val)
{
    switch (val)
    {
    case 'K':
        add_chesspiece(make_shared<King>(i, j, true, this, 'K'));
        break;
    case 'P':
        add_chesspiece(make_shared<Pawn>(i, j, true, this, 'P'));
        break;
    case 'R':
        add_chesspiece(make_shared<Rook>(i, j, true, this, 'R'));
        break;
    case 'N':
        add_chesspiece(make_shared<Knight>(i, j, true, this, 'N'));
        break;
    case 'B':
        add_chesspiece(make_shared<Bishop>(i, j, true, this, 'B'));
        break;
    case 'Q':
        add_chesspiece(make_shared<Queen>(i, j, true, this, 'Q'));
        break;
    case 'k':
        add_chesspiece(make_shared<King>(i, j, false, this, 'k'));
        break;
    case 'p':
        add_chesspiece(make_shared<Pawn>(i, j, false, this, 'p'));
        break;
    case 'r':
        add_chesspiece(make_shared<Rook>(i, j, false, this, 'r'));
        break;
    case 'n':
        add_chesspiece(make_shared<Knight>(i, j, false, this, 'n'));
        break;
    case 'b':
        add_chesspiece(make_shared<Bishop>(i, j, false, this, 'b'));
        break;
    case 'q':
        add_chesspiece(make_shared<Queen>(i, j, false, this, 'q'));
        break;
    case '.':
        break;
    default:
        break;
    }
}

void ChessBoard::movePiece(ChessMove chess_move)
{
    // In my implementation the x and y:s are swapped
    // so that the board behaves like a matrix.

    auto shared_piece = m_state(chess_move.from_y, chess_move.from_x);
    m_state(chess_move.from_y, chess_move.from_x) = nullptr;
    remove_piece(chess_move.to_y, chess_move.to_x);

    m_state(chess_move.to_y, chess_move.to_x) = shared_piece;
    shared_piece->m_y = chess_move.to_x;
    shared_piece->m_x = chess_move.to_y;
}

void ChessBoard::remove_piece(int x, int y)
{   
    auto shared_piece = m_state(x, y);
    if(shared_piece != nullptr)
    {
        if(shared_piece->is_white() == true)
        {
            m_white_pieces.erase(std::remove(m_white_pieces.begin(), m_white_pieces.end(), &(*shared_piece)), m_white_pieces.end());
        }
        else if(shared_piece->is_white() == false)
        {
            m_black_pieces.erase(std::remove(m_black_pieces.begin(), m_black_pieces.end(), &(*shared_piece)), m_black_pieces.end());
        }
    }
}

vector<ChessMove> ChessBoard::capturingMoves(bool is_white)
{
    vector<ChessMove> moves;
    vector<ChessPiece *> pieces = m_white_pieces;

    if (!is_white)
        pieces = m_black_pieces;

    for (ChessPiece *cp : pieces)
    {
        vector<ChessMove> cp_moves = cp->capturingMoves();
        moves.insert(moves.end(), cp_moves.begin(), cp_moves.end());
    }
    return moves;
}

vector<ChessMove> ChessBoard::nonCapturingMoves(bool is_white)
{
    vector<ChessMove> moves;
    vector<ChessPiece *> pieces = m_white_pieces;

    if (!is_white)
        pieces = m_black_pieces;

    for (ChessPiece *cp : pieces)
    {
        vector<ChessMove> cp_moves = cp->nonCapturingMoves();
        moves.insert(moves.end(), cp_moves.begin(), cp_moves.end());
    }
    return moves;
}

int ChessBoard::get_row_length()
{
    return row_length;
}

ChessBoard &operator>>(istream &is, ChessBoard &cb)
{
    std::string s(std::istreambuf_iterator<char>(is), {});

    int offset = 0;
    int row_length = cb.get_row_length();

    for(int j = 0; j < s.length(); j++)
    {
        if(s[j] == '\n')
        {
            offset++;
            continue;
        }
        char val = s[j];
        int y = (j - offset) / row_length;
        int x = (j - offset) % row_length;
        cb.insert_piece(y, x, val);
    }
    return cb;
}

ChessBoard &operator<<(ostream &os, ChessBoard &cb)
{
    int row_length = cb.get_row_length();
    for (int i = 0; i < row_length; i++)
    {
        for (int j = 0; j < row_length; j++)
        {
            std::shared_ptr<ChessPiece> cp = cb.getPiece(i, j);
            if (cp != nullptr)
                cout << cp.get()->get_sign();
            else
                cout << ".";
        }
        cout << endl;
    }
    return cb;
}

shared_ptr<ChessPiece> ChessBoard::getPiece(int x, int y)
{
    return m_state(x, y);
}

bool ChessBoard::check_promotion(ChessMove cm)
{
    char sign = cm.piece->latin1Representation();
    int to_y = cm.to_y;
    if (sign == 'P' && to_y == 0)
        return true;
    else if (sign == 'p' && to_y == 7)
        return true;
    return false;
}

void ChessBoard::promote_pawn(ChessMove cm, char sign)
{
    remove_piece(cm.from_y, cm.from_x);
    m_state(cm.from_y, cm.from_x) = nullptr;
    insert_piece(cm.to_y, cm.to_x, sign);
}

bool ChessBoard::try_promote_pawn(ChessMove cm, char sign)
{
    auto temp = m_state(cm.from_y, cm.from_x);
    m_state(cm.from_y, cm.from_x) = nullptr;
    insert_piece(cm.to_y, cm.to_x, sign);

    if(m_state(cm.to_y, cm.to_x)->capturingMoves().size() == 0)
    {
        // Lägg in den gamla pjäsen igen temporärt för att remove_piece ska fungera
        m_state(cm.from_y, cm.from_x) = temp;
        remove_piece(cm.from_y, cm.from_x);
        m_state(cm.from_y, cm.from_x) = nullptr;
        return true;   
    }
    else 
    {
        remove_piece(cm.to_y, cm.to_x);
        m_state(cm.from_y, cm.from_x) = temp;
        m_state(cm.to_y, cm.to_y) = nullptr;
        return false;
    }
}

bool ChessBoard::step_ahead_promotion(ChessMove cm, bool is_white)
{
    vector<char> pieces;

    if (is_white)
        pieces = {'Q', 'R', 'B', 'N'};
    else
        pieces = {'q', 'r', 'b', 'n'};

    
    for (char piece : pieces)
    {
        if(try_promote_pawn(cm, piece))
            return true;
    }
    return false;
}

void ChessBoard::reverse_pawn_promotion(ChessMove cm)
{
    m_state(cm.to_y, cm.to_x) = nullptr;
    insert_piece(cm.from_y, cm.from_x, cm.piece->latin1Representation());
}

void ChessBoard::print_white_pieces()
{
    for (ChessPiece *cp : m_white_pieces)
    {
        cout << cp->latin1Representation() << " : (" << cp->y() << ", " << cp->x() << ")" << endl;
    }
}

void ChessBoard::print_black_pieces()
{
    for (ChessPiece *cp : m_black_pieces)
    {
        cout << cp->latin1Representation() << " : (" << cp->y() << ", " << cp->x() << ")" << endl;
    }
}

vector<ChessPiece *> ChessBoard::get_white_pieces()
{
    return m_white_pieces;
}

vector<ChessPiece *> ChessBoard::get_black_pieces()
{
    return m_black_pieces;
}
