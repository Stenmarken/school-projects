#include <stdexcept>
#include <iostream>

// .-------------------------------------------------------
// |        count_if_followed_by (data, len, a, b);
// |
// | About: Returns the number of occurances of
// |        a followed by b in the range [data, data+len).
// '-------------------------------------------------------

int count_if_followed_by(char const *p, int len, char a, char b)
{

  int count = 0;
  char const *end = p + len;

  while (p != end)
  {
    //std::cout << "p + 1 : " << *(p + 1) << std::endl;
    //std::cout << "end : " << *end << std::endl;
    if (*p == a && *(p + 1) == b)
    {
      count += 1;
      if (++p == end)
        throw std::out_of_range("blah");
    }

    ++p;
  }

  return count;
}