#include <iostream>

int main()
{
    double a = 0.9;
    double b = 0.0;
    bool c = a > b;
    if(c == true)
        std::cout << "a är större än b" << std::endl;
    else
        std::cout << "a är inte större än b" << std::endl;
}