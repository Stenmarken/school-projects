
#include "ChessBoard.h"
#include <iostream>
#include <sstream>
#include "ChessPiece.h"
#include "Pawn.h"
#include "Queen.h"

using std::cout;
using std::endl;
using std::stringstream;
using std::vector;

int main(int argc, char *argv[])
{
    stringstream s;
    ChessBoard chess(8);

    s << ".....B.." << endl;
    s << "........" << endl;
    s << "........" << endl;
    s << "........" << endl;
    s << "........" << endl;
    s << "........" << endl;
    s << "........" << endl;
    s << ".N......" << endl;

    s >> chess;
    cout << chess;
    //int x, int y, bool is_white, ChessBoard * board, char sign
    Queen q(0, 0, true, &chess, 'Q');
    q.unnecessary_int = 10;
    return 0;
}


