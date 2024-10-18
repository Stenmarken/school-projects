#include "Node.h"
#include "helpers.h"
#include "Node.h"
#include <iostream>
#include <typeinfo>

using namespace std;
template <class S, class T>
void print_inorder(Node<S, T> *p)
{
    /*
    if (p != nullptr)
    {
        print_inorder(p->left);
        cout << "key : " << p->key << ". data : " << p->data << endl;
        print_inorder(p->right);
    }
    */
}

void test_tree()
{

    /*

                    20
                   /   \
                  15     25
                   \    /    \
                   17   22   29
                  /  \
                 16  19
    */

    Node<int, int> *root = new Node<int, int>(20, 20);
    Node<int, int> *n_25 = new Node<int, int>(25, 25, root);
    insert(root, n_25);

    Node<int, int> *n_15 = new Node<int, int>(15, 15, root);
    insert(root, n_15);

    Node<int, int> *n_17 = new Node<int, int>(17, 17, n_15);
    insert(root, n_17);
    
    Node<int, int> *n_16 = new Node<int, int>(16, 16, n_17);
    insert(root, n_16);
    
    Node<int, int> *n_19 = new Node<int, int>(19, 19, n_17);
    insert(root, n_19);

    Node<int, int> *n_22 = new Node<int, int>(22, 22, n_25);
    insert(root, n_22);
    
    Node<int, int> *n_29 = new Node<int, int>(29, 29, n_25);
    insert(root, n_29);

    
    for (auto p = root->begin(); p.has_next_node() == true; p.next_node())
    {
        cout << p() << endl;
    }

    cout << endl;
    for (auto j = root->begin(); j != root->end(); ++j)
    {
        cout << *j << endl;
    }
    
    
}

void print_three_times()
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
    Node<int, int> *root = new Node<int, int>(20, 20);
    // print_inorder(root);

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

    // second print
    for (auto p = root->begin(); p != root->end(); ++p)
    {
        cout << *p << endl;
    }
    cout << endl;
    // third print (exactly the same code as above)
    for (auto q = root->begin(); q != root->end(); ++q)
    {
        cout << *q << endl;
    }
    cout << endl;
    for (auto p = root->rbegin(); p != root->rend(); ++p)
    {
        cout << *p << endl;
    }
}

/*
int main()
{
    print_three_times();
    //test_tree();
    
    return 0;
}
*/


