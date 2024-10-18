#include <iostream>

template <int p, int i> struct check_if_prime
{
   typedef typename std::conditional<p % i == 0, std::false_type,
           typename check_if_prime<p, i-1>::type>::type type;
};

template <int p>
struct check_if_prime<p, 1> : public std::true_type
{
};

template <unsigned int X> 
struct is_prime
{
    static const bool value = check_if_prime<X, X-1>::type::value;
};

int main()
{
    bool a = is_prime<1000000000>::value;
    std::cout << a << std::endl;
}