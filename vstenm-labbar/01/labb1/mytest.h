#include <cxxtest/TestSuite.h>
#include "count_if_followed_by.h"

class MyTestSuite : public CxxTest::TestSuite 
{
public:
  void testCount_if_followed_by( void )
  {
    char const data[4] = {'a','b','a','b'};
    int  const result  = count_if_followed_by (data, 3, 'b', 'a');
    
    TS_ASSERT( result == 1 );
  }
  #
};