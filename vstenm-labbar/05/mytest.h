
// mytest.h
#include <cxxtest/TestSuite.h>
#include "bintree.h"

class MyTestSuite : public CxxTest::TestSuite
{
public:
  void testBiggerTree( void )
  {
    /*
                        
                        20 
                       /   \ 
                      15     25
                     / \    /    \ 
                    10 17   22   29
                   /  /  \ 
                  5  16  19
    */

    Node *p = new Node(20, 10);
    insert(p, 15, 15);
    insert(p, 10, 10);
    insert(p, 5, 5);
    insert(p, 17, 17);
    insert(p, 16, 16);
    insert(p, 19, 19);
    insert(p, 25, 25);
    insert(p, 22, 22);
    insert(p, 29, 29);

    TS_ASSERT(size(p) == 10);
    TS_ASSERT(is_balanced(p) == true);
    TS_ASSERT(max_height(p) == 4);
    TS_ASSERT(min_height(p) == 3);
    TS_ASSERT(find(p, 19) == 19);

    insert(p, 30, 2.5);
    TS_ASSERT(find(p, 30) == 2.5);

    remove(p, 19);
    TS_ASSERT_THROWS(find(p, 19), std::out_of_range);
    TS_ASSERT_THROWS(edit(p, 19), std::out_of_range);
    delete_tree(p);

  }
  void testEmptyTree( void )
  {
    TS_ASSERT(size(nullptr) == 0);
    TS_ASSERT(is_balanced(nullptr) == true);
    TS_ASSERT(max_height(nullptr) == 0);
    TS_ASSERT(min_height(nullptr) == 0);
  }
  void testLeaf( void )
  {
    Node *p = new Node(1, 1);
    TS_ASSERT(size(p) == 1);
    TS_ASSERT(is_balanced(p) == true);
    TS_ASSERT(max_height(p) == 1);
    TS_ASSERT(min_height(p) == 1);
    
    insert(p, 2, 2);
    TS_ASSERT(size(p) == 2);
    
    int found = find(p, 2);
    TS_ASSERT(found == 2);

    double& ref = edit(p, 2);
    ref = 3;
    TS_ASSERT(find(p, 2) == 3);

    remove(p, 1);
    TS_ASSERT(size(p) == 1);
    
    delete_tree(p);
    TS_ASSERT(size(p) == 0);
  }
};

