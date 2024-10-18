#include "bintree.h"
#include <queue>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <string>

// Print function taken from 
// https://stackoverflow.com/questions/36802354/print-binary-tree-in-a-pretty-way-using-c
void printBT(const std::string& prefix, const Node* node, bool isLeft)
{
    if( node != nullptr )
    {
        std::cout << prefix;

        std::cout << (isLeft ? "├──" : "└──" );

        // print the value of the node
        std::cout << node->key << std::endl;

        // enter the next tree level - left and right branch
        printBT( prefix + (isLeft ? "│   " : "    "), node->left, true);
        printBT( prefix + (isLeft ? "│   " : "    "), node->right, false);
    }
}
// Part of the print function taken from 
// https://stackoverflow.com/questions/36802354/print-binary-tree-in-a-pretty-way-using-c
void printBT(const Node* node)
{
    std::cout << std::endl;
    printBT("", node, false);    
}


// Insert code is based on the pseudocode from this article
// https://en.wikipedia.org/wiki/Binary_search_tree#Insertion
void insert(Node * & p, int key, double to_be_inserted)
{
    Node * z = new Node(key, to_be_inserted);

    Node * y = nullptr;
    Node * x = p;

    while(x != nullptr)
    {
        if (x->key == key)
        {
            x->data = to_be_inserted;
            return;
        }
        else if(x->key < key)
        {
            y = x;
            x = x->right;
        }
        else
        {
            y = x;
            x = x->left;
        }
    }
    if (z->key < y->key)
        y->left = z;
    else
        y->right = z;
}

const double & find(Node * p, const int & to_be_found)
{
    Node * y = nullptr;
    Node * & x = p;

    while(x != nullptr)
    {
        y = x;
        if(to_be_found < x->key)
            x = x->left;
        else if(to_be_found > x->key)
            x = x->right;
        else 
        {
            return x->data;
        }
    }
    throw std::out_of_range("The key was not found in the tree!");
}

double & edit(Node * p, const int & to_be_changed)
{
    Node * y = nullptr;
    Node * & x = p;

    while(x != nullptr)
    {
        y = x;
        if(to_be_changed < x->key)
            x = x->left;
        else if(to_be_changed > x->key)
            x = x->right;
        else 
        {
            return x->data;
        }
    } 
  throw std::out_of_range("The key was not found in the tree!");
}

unsigned int size(Node * p)
{
    if (p == nullptr)
        return 0;
    else
        return size(p->left) + size(p->right) + 1;
}

unsigned int min_height(Node * p)
{
    if (p == nullptr)
        return 0;
    else
        return std::min(min_height(p->left), min_height(p->right)) + 1;
}

unsigned int max_height(Node * p)
{
    if (p == nullptr)
        return 0;
    else
        return std::max(max_height(p->left), max_height(p->right)) + 1;
}

bool is_balanced(Node * p)
{
    if (p == nullptr)
    {
        return true;
    }
    else
    {
        int left = max_height(p->left);
        int right = max_height(p->right);
        return abs(left - right) <= 1 && is_balanced(p->left) && is_balanced(p->right);
    }
}

void remove(Node * & p, const int & key)
{
    if (p == nullptr)
        return;
    if(key < p->key)
        return remove(p->left, key);
    else if (key > p->key)
        return remove(p->right, key);
    else
    {
        if (p->left == nullptr && p->right == nullptr)
        {
            delete p;
            p = nullptr;
            return;
        }
        else if (p->left == nullptr)
        {
            Node *temp = p;
            p = p->right;
            delete temp;
            return;
        }
        else if (p->right == nullptr)
        {
            Node *temp = p;
            p = p->left;
            delete temp;
            return;
        }
        else 
        {
            Node *pred = inorder_predecessor(p);
            p->key = pred->key;
            p->data = pred->data;
            remove(p->left, pred->key);
        }
    }
}


Node * inorder_predecessor(Node * p)
{
    p = p->left;
    while(p->right != nullptr)
        p = p->right;
    return p;
}

void delete_tree(Node * & p)
{
    delete_tree_helper(p);
    p = nullptr;
}

void delete_tree_helper(Node * & p)
{
    if (p == nullptr)
        return;
    delete_tree(p->left);
    delete_tree(p->right);
    delete p;
}

/*
int main()
{


}
*/

