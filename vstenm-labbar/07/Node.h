
#ifndef NODE_H
#define NODE_H

#include <stdio.h>
#include <iterator>
#include <algorithm>
#include "Node_iterator.h"
#include "Const_node_iterator.h"

template <class S, class T>
struct Node
{
    typedef real_node_iterator<S, T> iterator;       // TODO
    typedef const_real_node_iterator<S, T> const_iterator; // TODO
public:
    S key;
    T data;
    Node<S, T> *parent; // points to parent node
    Node<S, T> *right;
    Node<S, T> *left;

    Node()
    {
        parent = nullptr;
        right = nullptr;
        left = nullptr;
    }

    Node(S key_data, T value_data)
    {
        key = key_data;
        data = value_data;
        parent = nullptr;
        right = nullptr;
        left = nullptr;
    }
    
    Node(S key_data, T value_data, Node<S,T> * parent_node)
    {
        key = key_data;
        data = value_data;
        parent = parent_node;
        right = nullptr;
        left = nullptr;
    }

    iterator begin()
    {
        Node<int, int> *current = this;
        while (current->left != nullptr)
            current = current->left;
        return iterator(current, this);
    }
    iterator end()
    {
        Node<int, int> *current = this;
        while (current->right != nullptr)
            current = current->right;
        current->right = nullptr;
        return iterator(current->right, this);
    }
    
    
    const_iterator cbegin()
    {
        Node<int, int> *current = this;
        while (current->left != nullptr)
            current = current->left;
        return const_iterator(current, false, true, this);
    }
    const_iterator cend()
    {
        Node<int, int> *current = this;
        while (current->right != nullptr)
            current = current->right;
        return const_iterator(current->right, false, true, this);
    }
    

    iterator rbegin()
    {
        Node<S, T> *current = this;
        while (current->right != nullptr)
            current = current->right;
        return iterator(current, true, false, this);
    }
    iterator rend()
    {
        Node<int, int> *current = this;
        while (current != nullptr)
            current = current->left;
        return iterator(current, true, false, this);
    }
};
#endif 
