#ifndef HELPERS_H
#define HELPERS_H
#include <iostream>
#include <string>
//#include "Node.h"

// Print function taken from 
// https://stackoverflow.com/questions/36802354/print-binary-tree-in-a-pretty-way-using-c
template<class S, class T>
void printBT(const std::string& prefix, const Node<S, T> *node, bool isLeft)
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
template<class S, class T>
void printBT(const Node<S, T> *node)
{
    std::cout << std::endl;
    printBT("", node, false);    
}

// Insert code is based on the pseudocode from this article
// https://en.wikipedia.org/wiki/Binary_search_tree#Insertion
template<class S, class T>
void insert(Node<S, T> * & p, S key, T to_be_inserted)
{
    Node<S, T> * z = new Node(key, to_be_inserted);

    Node<S, T> * y = nullptr;
    Node<S, T> * x = p;

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

// Insert code is based on the pseudocode from this article
// https://en.wikipedia.org/wiki/Binary_search_tree#Insertion
template<class S, class T>
void insert(Node<S, T> * & p, Node<S, T> * & z)
{

    Node<S, T> * y = nullptr;
    Node<S, T> * x = p;

    while(x != nullptr)
    {
        if (x->key == z->key)
        {
            x->data = z->data;
            return;
        }
        else if(x->key < z->key)
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

template<class S, class T>
const double & find(Node<S, T> *p, const T & to_be_found)
{
    Node<S, T> * y = nullptr;
    Node<S, T> * & x = p;

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

template<class S, class T>
double & edit(Node<S, T> *p, const T & to_be_changed)
{
    Node<S, T> * y = nullptr;
    Node<S, T> * & x = p;

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

template<class S, class T>
unsigned int size(Node<S, T> *p)
{
    if (p == nullptr)
        return 0;
    else
        return size(p->left) + size(p->right) + 1;
}

template<class S, class T>
unsigned int min_height(Node<S, T> * p)
{
    if (p == nullptr)
        return 0;
    else
        return std::min(min_height(p->left), min_height(p->right)) + 1;
}

template<class S, class T>
unsigned int max_height(Node<S, T> *p)
{
    if (p == nullptr)
        return 0;
    else
        return std::max(max_height(p->left), max_height(p->right)) + 1;
}

template<class S, class T>
bool is_balanced(Node<S, T> * p)
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

template<class S, class T>
void remove(Node<S, T> * & p, const S & key)
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
            Node<S, T> *temp = p;
            p = p->right;
            delete temp;
            return;
        }
        else if (p->right == nullptr)
        {
            Node<S, T> *temp = p;
            p = p->left;
            delete temp;
            return;
        }
        else 
        {
            Node<S, T> *pred = inorder_predecessor(p);
            p->key = pred->key;
            p->data = pred->data;
            remove(p->left, pred->key);
        }
    }
}

template<class S, class T>
Node<S, T> * inorder_predecessor(Node<S, T> * p)
{
    p = p->left;
    while(p->right != nullptr)
        p = p->right;
    return p;
}

template<class S, class T>
void delete_tree(Node<S, T> * & p)
{
    delete_tree_helper(p);
    p = nullptr;
}

template<class S, class T>
void delete_tree_helper(Node<S, T> * & p)
{
    if (p == nullptr)
        return;
    delete_tree(p->left);
    delete_tree(p->right);
    delete p;
}
#endif