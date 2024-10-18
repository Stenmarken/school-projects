// mytest.h
#include <cxxtest/TestSuite.h>
#include "count_if_followed_by.h"

class MyTestSuite : public CxxTest::TestSuite
{
public:
    void testCount_if_followed_by(void)
    {
        char const data[4] = {'a', 'b', 'a', 'b'};

        int const result = count_if_followed_by(data, 3, 'b', 'a');

        TS_ASSERT(result == 1);
    }

    void testFirst_element(void)
    {
        // Correct behavior
        char const data[4] = {'a', 'b', 'a', 'b'};

        int const result = count_if_followed_by(data, 1, 'b', 'a');

        TS_ASSERT(result == 0);
    }

    void testLast_element(void)
    {
        // Incorrect behavior
        char const data[4] = {'a', 'b', 'a', 'b'};
        int const result = count_if_followed_by(data, 4, 'a', 'b');

        // THE FOLLOWING IS WRONG
        TS_ASSERT(result == 2);
    }

    void testEmpty_list(void)
    {
        // Correct behavior
        char const data[0] = {};

        int const result = count_if_followed_by(data, 1, 'X', 'G');

        TS_ASSERT(result == 0);
    }
};