#include <iostream>
#include <cmath>
#include <vector>
#include <sstream>
#include <string>

using namespace std;

vector<int> find_primes(int N)
{
    int *numbers = new int[N]; // Default värde för värden i en int array är 0
                               // 0 motsvarar ett primtal, 1 motsvarar inte ett icke-primtal
    vector<int> primes;

    for(int i = 2; i < sqrt(N); i++)
    {
        if (numbers[i] == 0)
        {
            for(int j = i*i; j < N; j += i)
                numbers[j] = 1;
        }
    }
    
    for(int i = 2; i < N; i++)
    {
        if (numbers[i] == 0)
            primes.push_back(i);
    }
    
    delete [] numbers;

    return primes;
}

vector<int> basic_algorithm(int N)
{
    /*
    Begin by assigning a vector one or more prime numbers from 2 and upwards. 
    Iterate from 2 or your highest prime number to N. In each iteration, do an inner 
    loop that checks if the number is divisible with your current known prime numbers, 
    in which case it is not a prime number. The inner loop needs only check 
    divisibility up to the square root of the number.
    */

    vector<int> primes;
    primes.push_back(2);

    for(int i = 2; i < N; i++)
    {
        bool skip = false;
        for(int j = 0; j < primes.size(); j++)
        {
            if (primes.at(j) > sqrt(N))
                break;
            if (i % primes.at(j) == 0)
            {
                skip = true;
                break;
            }
        }
        if (!skip)
            primes.push_back(i);
    }
    return primes;
}

std::string hard_coded_primes(int n)
{
    stringstream ss;
    ss << "vector<int> primes = {";
    vector<int> v = find_primes(n);
    for (int prime : v)
    {
        ss << prime << ", ";
    }
    ss << "};";
    return ss.str();
}

int main()
{
    vector<int> v = find_primes(1000);
    vector<int> v_basic = basic_algorithm(1000);

    cout << "Length of v : " << v.size() << endl;
    cout << "Length of v_basic : " << v_basic.size() << endl; 
    cout << "Hard coded primes : " << hard_coded_primes(1000) << endl;
    
    return 0;
}