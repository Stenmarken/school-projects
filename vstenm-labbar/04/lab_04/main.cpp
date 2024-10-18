//
// DD1388 - Lab 4: Losing Chess
//
#include "ChessBoard.h"
#include "ChessPiece.h"
#include <iostream>
#include <sstream>
#include <iterator>
#include <random>

// Implement additional functions or classes of your choice
using namespace std;

std::random_device r;
std::default_random_engine e1(r());

ChessMove select_randomly(vector<ChessMove> moves)
{

    std::uniform_int_distribution<int> uniform_dist(0, moves.size() - 1);
    int mean = uniform_dist(e1);
    return moves.at(mean);
}

char select_random_piece(bool is_white)
{
    std::uniform_int_distribution<int> uniform_dist(0, 3);
    int mean = uniform_dist(e1);
    vector<char> pieces = {'q', 'r', 'b', 'n'};
    char piece = pieces.at(mean);
    if (is_white)
    {
        piece = toupper(piece);
    }
    return piece;
}

bool random_thinker(ChessBoard &cb, bool is_white)
{
    vector<ChessMove> non_capturing = cb.nonCapturingMoves(is_white);
    vector<ChessMove> capturing = cb.capturingMoves(is_white);

    if (capturing.size() == 0 && non_capturing.size() == 0)
        return false;

    if (capturing.size() == 0)
    {
        ChessMove cm = select_randomly(non_capturing);
        if (cb.check_promotion(cm))
        {
            cb.promote_pawn(cm, select_random_piece(is_white));
            return true;
        }
        cb.movePiece(cm);
        return true;
    }

    ChessMove cm = select_randomly(capturing);
    if (cb.check_promotion(cm))
    {
        cb.remove_piece(cm.to_y, cm.to_x);
        cb.promote_pawn(cm, select_random_piece(is_white));
        return true;
    }
    cb.movePiece(cm);
    return true;
}

bool step_ahead(ChessBoard &cb, bool is_white)
{
    vector<ChessMove> non_capturing = cb.nonCapturingMoves(is_white);
    vector<ChessMove> capturing = cb.capturingMoves(is_white);

    if (non_capturing.size() == 0 && capturing.size() == 0)
        return false;

    if (capturing.size() == 0)
    {
        for (ChessMove cm : non_capturing)
        {
            if (cb.check_promotion(cm))
            {
                int num_capturing_before = cb.capturingMoves(!is_white).size();

                bool non_capturing_promotion = cb.step_ahead_promotion(cm, is_white);
                if (non_capturing_promotion && cb.capturingMoves(!is_white).size() >= num_capturing_before)
                    return true;

                int num_capturing_before_promotion = cb.capturingMoves(!is_white).size();
                cb.promote_pawn(cm, select_random_piece(is_white));
                bool force_capture = cb.capturingMoves(!is_white).size() >= num_capturing_before_promotion;
                if (force_capture)
                    return true;
                else
                    cb.reverse_pawn_promotion(cm);
            }
            else
            {
                int num_capturing_before2 = cb.capturingMoves(!is_white).size();
                cb.movePiece(cm);
                bool force_capture = cb.capturingMoves(!is_white).size() > num_capturing_before2;

                if (force_capture)
                    return true;
                else
                    cb.movePiece(ChessMove(cm.to_x, cm.to_y, cm.from_x, cm.from_y, cm.piece));
            }
        }
        cb.movePiece(select_randomly(non_capturing));
        return true;
    }
    else
    {
        for (ChessMove cm : capturing)
        {
            int num_capturing_before = cb.capturingMoves(!is_white).size();
            cb.movePiece(cm);
            int num_capturing_after = cb.capturingMoves(!is_white).size();
            bool force_capture = num_capturing_after > num_capturing_before;
            if (force_capture)
                return true;
            else
                cb.movePiece(ChessMove(cm.to_x, cm.to_y, cm.from_x, cm.from_y, cm.piece));
        }
        cb.movePiece(select_randomly(capturing));
        return true;
    }
    return true;
}

bool check_infinte_bishops(ChessBoard &board)
{
    if (board.get_white_pieces().size() == 1 && board.get_black_pieces().size() == 1)
    {
        if (board.get_white_pieces()[0]->get_sign() == 'B' && board.get_black_pieces()[0]->get_sign() == 'b')
            return true;
    }
    return false;
}

int black_starts_game(bool white_is_random, bool black_is_random, ChessBoard &chess)
{
    int won = 0;
    while (true)
    {
        if (check_infinte_bishops(chess))
        {
            cout << "Infinite bishops! Draw!" << endl;
            won = 0;
            break;
        }

        if (black_is_random)
        {
            if (!random_thinker(chess, false))
            {
                cout << "Black won!" << endl;
                won = 2;
                break;
            }
        }
        else
        {
            if (!step_ahead(chess, false))
            {
                cout << "Black won!" << endl;
                won = 2;
                break;
            }
        }

        cout << chess;
        cout << endl;

        if (white_is_random)
        {
            if (!random_thinker(chess, true))
            {
                cout << "White won!" << endl;
                won = 1;
                break;
            }
        }
        else
        {
            if (!step_ahead(chess, true))
            {
                cout << "White won!" << endl;
                won = 1;
                break;
            }
        }
        cout << chess;
        cout << endl;
    }
    return won;
}

int white_starts_game(bool white_is_random, bool black_is_random, ChessBoard &chess)
{
    // 0 - draw
    // 1 - white won
    // 2 - black won
    int won = 0;

    while (true)
    {
        if (check_infinte_bishops(chess))
        {
            cout << "Infinite bishops! Draw!" << endl;
            won = 0;
            break;
        }

        if (white_is_random)
        {
            if (!random_thinker(chess, true))
            {
                cout << "White won!" << endl;
                won = 1;
                break;
            }
        }
        else
        {
            if (!step_ahead(chess, true))
            {
                cout << "White won!" << endl;
                won = 1;
                break;
            }
        }
        cout << chess;
        cout << endl;

        if (black_is_random)
        {
            if (!random_thinker(chess, false))
            {
                cout << "Black won!" << endl;
                won = 2;
                break;
            }
        }
        else
        {
            if (!step_ahead(chess, false))
            {
                cout << "Black won!" << endl;
                won = 2;
                break;
            }
        }
        cout << chess;
        cout << endl;
    }
    return won;
}


int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cout << "The program needs three arguments" << endl;
        cout << "Firstly -W or -B for who starts the game" << endl;
        cout << "Then you need to choose the type of AI for white and black" << endl;
        cout << "-r is for random AI and -s is for one step ahead AI" << endl;
        cout << "Example prompt: ./.output -W -r -r" << endl;
        return 1;
    }

    bool white_is_random = false;
    bool black_is_random = false;
    bool white_starts = false;

    if (strcmp(argv[1], "-W") == 0)
        white_starts = true;
    else if (strcmp(argv[1], "-B") == 0)
        white_starts = false;
    else
    {
        cout << "The program needs two arguments. One for the white AI and one for the black AI." << endl;
        cout << "-r for random AI and -s for one step ahead" << endl;
        return 1;
    }

    if (strcmp(argv[2], "-r") == 0)
        white_is_random = true;
    else if (strcmp(argv[2], "-s") == 0)
        white_is_random = false;
    else
    {
        cout << "The program needs two arguments. One for the white AI and one for the black AI." << endl;
        cout << "-r for random AI and -s for one step ahead" << endl;
        return 1;
    }

    if (strcmp(argv[3], "-r") == 0)
        black_is_random = true;
    else if (strcmp(argv[3], "-s") == 0)
        black_is_random = false;
    else
    {
        cout << "The program needs two arguments. One for the white AI and one for the black AI." << endl;
        cout << "-r for random AI and -s for one step ahead" << endl;
        return 1;
    }
    

    int won = 0;
    int white_wins = 0;
    int black_wins = 0;
    
    ChessBoard chess(8);
    stringstream s;

    s << "rnbqkbnr" << endl;
    s << "pppppppp" << endl;
    s << "........" << endl;
    s << "........" << endl;
    s << "........" << endl;
    s << "........" << endl;
    s << "PPPPPPPP" << endl;
    s << "RNBQKBNR" << endl;

    s >> chess;

    if (white_starts)
        won = white_starts_game(white_is_random, black_is_random, chess);
    else
        won = black_starts_game(white_is_random, black_is_random, chess);

    return 0;
}

