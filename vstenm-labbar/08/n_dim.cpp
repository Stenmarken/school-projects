#include <vector>
#include <iostream>

using namespace std;


// Inspiration till NdimMatrix är tagen från 
// https://www.reddit.com/r/learnprogramming/comments/bnfui0/c_template_meta_programming_ndimensional_stdvector/
template <int dimension, typename T>
struct NdimMatrix
{
    vector<NdimMatrix<dimension - 1, T>> vec;

    NdimMatrix(){}   

    NdimMatrix(int length)
    {
        vec = vector<NdimMatrix<dimension - 1, T>>(length);
        for (auto &elem : vec)
            elem = NdimMatrix<dimension - 1, T>(length);
    } 

    NdimMatrix<dimension - 1, T> &operator[](int index)
    {
        return vec.at(index);
    }
};

template <typename T>
struct NdimMatrix<1, T>
{
    vector<T> vec;

    NdimMatrix(){}

    NdimMatrix(int length)
    {
        vec = vector<T>(length);
    }

    T &operator[](int index)
    {
        return vec.at(index);
    }
};

int main()
{
    NdimMatrix<3, double> n(9.0); // a cube with 9 * 9 * 9 elements
    n[1][2][3] = 5.0;
    NdimMatrix<6, int> m(5); // a matrix in six dimensions with 5 * 5 * 5 * 5 * 5 * 5 elements
    m[1][3][2][1][4][0] = 7;
    NdimMatrix<3, int> t(5);
    t = m[1][3][2];                                // assign part (slice) of m to t, the dimensions and element length matches
    t[1][4][0] = 2;                                // changes t but not m
    std::cout << m[1][3][2][1][4][0] << std::endl; // 7
    std::cout << t[1][4][0] << std::endl;          // 2
    
    return 0;
    
}