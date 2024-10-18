// mytest.h
#include <cxxtest/TestSuite.h>
#include "Node.h"
#include "helpers.h"
#include <string.h>

class MyTestSuite : public CxxTest::TestSuite
{
public:
    void testDereference(void)
    {
        Node<int, int> *real_root = new Node<int, int>;
        real_root->key = 5;
        real_root->data = 5;

        real_node_iterator<int, int> n_i2 = real_node_iterator<int, int>(real_root, real_root);
        *n_i2 = 25;
        TS_ASSERT_EQUALS(n_i2.get_key(), 5);
        TS_ASSERT_EQUALS(n_i2.get_data(), 25);
    }

    void testAssignment(void)
    {
        Node<int, int> *real_root = new Node<int, int>;
        real_root->key = 7;
        real_root->data = 7;

        Node<int, int> *new_real_root = new Node<int, int>;
        new_real_root->key = 9;
        new_real_root->data = 9;

        real_node_iterator<int, int> n_i2 = real_node_iterator<int, int>(real_root, real_root);
        n_i2 = real_node_iterator<int, int>(new_real_root, new_real_root);
        TS_ASSERT_EQUALS(n_i2.get_key(), 9);
        TS_ASSERT_EQUALS(n_i2.get_data(), 9);
    }

    void testEqualityInequality(void)
    {
        Node<int, int> *real_root = new Node<int, int>;
        real_root->key = 7;
        real_root->data = 7;

        Node<int, int> *new_real_root = new Node<int, int>;
        new_real_root->key = 9;
        new_real_root->data = 9;

        real_node_iterator<int, int> n_i2 = real_node_iterator<int, int>(real_root, real_root);
        real_node_iterator<int, int> n_i3 = real_node_iterator<int, int>(new_real_root, real_root);
        TS_ASSERT_EQUALS(n_i2 == n_i3, false);
        TS_ASSERT_EQUALS(n_i2 != n_i3, true);
    }

    void testIncrementOperators(void)
    {
        Node<int, int> *real_root = new Node<int, int>;
        real_root->key = 7;
        real_root->data = 7;

        Node<int, int> *right = new Node<int, int>;
        right->key = 9;
        right->data = 9;

        real_root->right = right;

        real_node_iterator<int, int> n_i = real_node_iterator<int, int>(real_root, real_root);
        ++n_i;
        TS_ASSERT_EQUALS(n_i.get_key(), 9);
        TS_ASSERT_EQUALS(n_i.get_data(), 9);
    }

    void testSwapOperator(void)
    {
        Node<int, int> *real_root = new Node<int, int>;
        real_root->key = 7;
        real_root->data = 7;

        Node<int, int> *right = new Node<int, int>;
        right->key = 9;
        right->data = 9;
        real_root->right = right;

        real_node_iterator n_1 = real_node_iterator<int, int>(real_root, real_root);
        real_node_iterator n_2 = real_node_iterator<int, int>(right, right);

        TS_ASSERT_EQUALS(n_1.get_key(), 7);
        TS_ASSERT_EQUALS(n_1.get_data(), 7);
        TS_ASSERT_EQUALS(n_2.get_key(), 9);
        TS_ASSERT_EQUALS(n_2.get_data(), 9);

        swap(n_1, n_2);
        TS_ASSERT_EQUALS(n_1.get_key(), 9);
        TS_ASSERT_EQUALS(n_1.get_data(), 9);
        TS_ASSERT_EQUALS(n_2.get_key(), 7);
        TS_ASSERT_EQUALS(n_2.get_data(), 7);
    }

    void testInorderIncrements(void)
    {
        Node<int, int> *root = new Node<int, int>(20, 20);

        Node<int, int> *n_25 = new Node<int, int>(25, 25, root);
        insert(root, n_25);

        Node<int, int> *n_15 = new Node<int, int>(15, 15, root);
        insert(root, n_15);

        Node<int, int> *n_10 = new Node<int, int>(10, 10, n_15);
        insert(root, n_10);

        Node<int, int> *n_5 = new Node<int, int>(5, 5, n_10);
        insert(root, n_5);

        Node<int, int> *n_22 = new Node<int, int>(22, 22, n_25);
        insert(root, n_22);

        Node<int, int> *n_29 = new Node<int, int>(29, 29, n_25);
        insert(root, n_29);

        Node<int, int> *n_17 = new Node<int, int>(17, 17, n_15);
        insert(root, n_17);

        Node<int, int> *n_16 = new Node<int, int>(16, 16, n_17);
        insert(root, n_16);

        Node<int, int> *n_19 = new Node<int, int>(19, 19, n_17);
        insert(root, n_19);

        std::stringstream ss;
        for (auto j = root->begin(); j != root->end(); ++j)
            ss << *j << " ";
        string compare_str = "5 10 15 16 17 19 20 22 25 29 ";
        TS_ASSERT_EQUALS(ss.str(), compare_str);

        std::stringstream ss2;
        for (auto k = root->rbegin(); k != root->rend(); ++k)
            ss2 << *k << " ";
        string compare_str2 = "29 25 22 20 19 17 16 15 10 5 ";
        TS_ASSERT_EQUALS(ss2.str(), compare_str2);
    }

    void testOperatorChaining(void)
    {
        Node<int, int> *root = new Node<int, int>(20, 20);

        Node<int, int> *n_25 = new Node<int, int>(25, 25, root);
        insert(root, n_25);

        Node<int, int> *n_27 = new Node<int, int>(27, 27, n_25);
        insert(root, n_27);
        auto p = root->begin();
        ++ ++p;
        TS_ASSERT_EQUALS(*p, 27);

        auto j = root->rbegin();
        ++ ++j;
        TS_ASSERT_EQUALS(*j, 20);

        auto i_it = root->begin();
        i_it++ ++;
        TS_ASSERT_EQUALS(*i_it, 27)

        auto r_it = root->rbegin();
        int val = *r_it++++;
        TS_ASSERT_EQUALS(val, 20);
    }

    void testConstIterator(void)
    {
        Node<int, int> *root = new Node<int, int>(20, 20);

        Node<int, int> *n_25 = new Node<int, int>(25, 25, root);
        insert(root, n_25);

        auto c_it_beg = root->cbegin();
        std::stringstream ss;
        ss << *c_it_beg;
        ++c_it_beg;
        ss << " " << *c_it_beg;
        TS_ASSERT_EQUALS(ss.str(), "20 25");

        //*c_it_beg = 30; // causes: error: assignment of read-only location ‘c_it_beg.const_real_node_iterator<int, int>::operator*()’
    }
};