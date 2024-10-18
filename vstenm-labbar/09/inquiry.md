# Inquiry

Write your answers below and include the questions. Do copy relevant code to your answers so that both questions and answers are in context 

## Answers


## What did you learn from this assignment?
I learned about considerations when making game desing, about constexpr and about the spaceship operator!

## What was the most difficult part of this assignment?
Removing the board state and implementing things just
with vectors wasn't in and of itself difficult but it meant
I had to solve a few annoying bugs.

## Why would you want this implementation 
There are a few good reasons for using this implementation. The most obvious one is the reduced complexity from not
having to check and alter the board state constantly. Another reason is reducing the amount of memory used. If we have
a board state we would most likely also want a second datastructure for all the pieces. With this implementation we only
need to store the pieces. Saving memory becomes important if the board becomes really big.

## For what types of games is it impractical/impossible to use a board layout?
I'm guessing that for big open world games it isn't feasible to use a board layout. In these cases the world
cannot be easily divided into squares where we can definitively say if a square is occupied or not. 


## What is your opinion of implementing loosing chess this way compared to the previous way?
In the beginning, it seemed like a really bad idea since using a matrix for a chessboard makes intuitive sense.
However, when I started to implement the code I noticed that the code became simpler this way. Earlier, when 
adding and removing pieces I had to both change the matrix and the vectors for the white and black pieces. 
The downside of the approach is of course the overhead every time you need to know what piece is on a certain square.
But that maybe isn't a big problem in chess where the number of pieces is limited.

## How many methods in std::array were made constexpr in c++17
If I'm interpreting the documentation correctly (https://en.cppreference.com/w/cpp/container/array), it looks like the following 9 methods were made constexpr in c++17
1. std::array<T,N>::at
2. std::array<T,N>::operator[]
3. std::array<T,N>::front
4. std::array<T,N>::back
5. std::array<T,N>::data
6. std::array<T,N>::begin, std::array<T,N>::cbegin
7. std::array<T,N>::end, std::array<T,N>::cend
8. std::array<T,N>::rbegin, std::array<T,N>::crbegin
9. std::array<T,N>::rend, std::array<T,N>::crend


## What does the new operator<=> in std::array do?
As is explained in this article https://stackoverflow.com/questions/47466358/what-is-the-spaceship-three-way-comparison-operator-in-c,
the operator<=> does the following:

auto cmp  = a <=> b;
1. if a > b then cmp > 0
2. if a == b then cmp = 0
3. if a < b then cmp < 0

Here is an example of the operator taken from the same article.

```
include <iostream>

using namespace std;

int main()
{
        int lhs = 10, rhs = 20;
        auto result = lhs <=> rhs;

        if (result < 0) {
                cout << "lhs is less than rhs" << endl;
        }
        else if (result > 0) {
                cout << "lhs is greater than rhs" << endl;
        }
        else {
                cout << "lhs and rhs are equal" << endl;
        }

}
```