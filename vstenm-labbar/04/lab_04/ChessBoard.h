//
// DD1388 - Lab 4: Losing Chess
//

#ifndef CHESSBOARD_H
#define CHESSBOARD_H

#include <vector>
#include <istream>
#include <memory>
#include "ChessMove.h"
#include "Matrix.h"   // Use the "-I ../02" flag to let the compiler find Matrix.h

using namespace std;

class ChessPiece;

class ChessBoard {
    // add additional members or functions of your choice

private:
    // Alternative 1 (the matrix owns the chess pieces):
    Matrix<shared_ptr<ChessPiece>> m_state; // Matrix from lab 2
    vector<ChessPiece *> m_white_pieces;
    vector<ChessPiece *> m_black_pieces;
    int row_length;

    // Alternative 2 (the vectors own the chess pieces):
    // Matrix<ChessPiece *> m_state; // Matrix from lab 2
    // vector<shared_ptr<ChessPiece>> m_white_pieces;
    // vector<shared_ptr<ChessPiece>> m_black_pieces;

public:
    ChessBoard();
    ChessBoard(int n);
    void movePiece(ChessMove chess_move);
    vector<ChessMove> capturingMoves(bool is_white);
    vector<ChessMove> nonCapturingMoves(bool is_white);
    shared_ptr<ChessPiece> getPiece(int x, int y);
    void insert_piece(int i, int j, char val);
    int get_row_length();
    bool check_promotion(ChessMove cm);
    void promote_pawn(ChessMove cm, char sign);
    void add_chesspiece(shared_ptr<ChessPiece> piece);
    bool step_ahead_promotion(ChessMove cm, bool is_white);
    void print_white_pieces();
    void print_black_pieces();
    void remove_piece(int x, int y);
    vector<ChessPiece *> get_white_pieces();
    vector<ChessPiece *> get_black_pieces();
    bool try_promote_pawn(ChessMove cm, char sign);
    void reverse_pawn_promotion(ChessMove cm);
};

ChessBoard & operator>>(istream & is, ChessBoard & cb);
ChessBoard & operator<<(ostream & os, ChessBoard & cb);

#endif //CHESSBOARD_H
