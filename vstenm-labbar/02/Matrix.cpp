#include "Matrix.h"
#include <stdio.h>
#include <initializer_list>
#include <vector>


using namespace std;

void somefunction()
{
    //int* m_vec = new int[9] {0,1,2,3,4,5,6,7,8};
    int *m_vec = new int[9];
    for (int i = 0; i < 9; i++)
        m_vec[i] = i;



    int old_capacity = 9;
    int *new_m_vec = new int[6];
    int col = 3;

    for(int i = 0; i < 3; i++)
    {
        for(int j = 1; j <= 3; j++)
        {
            if (j == col)
                continue;
            if (j < col)
                new_m_vec[i * 3 + j - 1] = m_vec[i * 3 + j - 1];
            else
                new_m_vec[i * 3 + j - 1] = m_vec[i * 3 + j - 2];
        }
    }


    std::cout << "\n" << std::endl;
    for (int i = 0; i < 6; i++)
    {
        std::cout << m_vec[i] << " ";
        if ((i + 1) % 3 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;

    delete[] m_vec;
    delete[] new_m_vec;
}
