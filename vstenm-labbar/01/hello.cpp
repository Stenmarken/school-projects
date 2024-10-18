#include <iostream>
#include "hello.h"

void hello(const char *name, int count)
{
    std::string s = "Hello, ";
    std::string space = " ";
    for (int i = 0; i < count; i++)
    {
        s += name + space;
    }
    s[s.size() - 1] = '!';
    if (count > 0)
        std::cout << s << std::endl;
}

std::pair<const char *, int> parse_args(int argc, char *argv[])
{
    std::string s;
    int parsed;
    switch (argc)
    {
    case 1:
        return std::make_pair("world", 1);
        break;
    case 2:
        return std::make_pair(argv[1], 1);
        break;
    case 3:
        s = argv[2];
        parsed = std::atoi(argv[2]);

        if ((parsed == 0 && s != "0") || parsed < 0)
        {
            std::cerr << "error: 2nd argument must be an integral greater than or equal zero!" << std::endl;
            return std::make_pair(argv[1], -1);
        }
        return std::make_pair(argv[1], parsed);
        break;
    default:
        std::cerr << "error: Too many arguments!" << std::endl;
        return std::make_pair(argv[1], -1);
        break;
    }
    return std::make_pair("", -1);
}