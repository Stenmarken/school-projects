
#ifndef FRIEND_NODE_H
#define FRIEND_NODE_H

#include <stdio.h>
#include <iterator>
#include <algorithm>
#include <iostream>
//#include "Friend_tests.h"

using namespace std;

template <class S, class T>
struct Node;

//class MyTestSuite;


// Regular helper functions

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

template <class S, class T>
void print_inorder(Node<S, T> *p)
{
    if (p != nullptr)
    {
        print_inorder(p->left);
        cout << p->data << endl;
        print_inorder(p->right);
    }
}

// Regular iterator

template <class S, class T>
class real_node_iterator
{
public:
    Node<S, T> *m_pointer;
    bool reversed;
    bool is_const;

    real_node_iterator(Node<S, T> *pointer)
    {
        m_pointer = pointer;
        reversed = false;
        is_const = false;
    }

    real_node_iterator(Node<S, T> *pointer, bool reversed, bool is_const)
    {
        m_pointer = pointer;
        this->reversed = reversed;
        this->is_const = is_const;
    }

    real_node_iterator(const real_node_iterator &other)
    {
        m_pointer = other.m_pointer;
        reversed = other.reversed;
        is_const = other.is_const;
    }

    real_node_iterator &operator=(const real_node_iterator &other)
    {
        m_pointer = other.m_pointer;
        return *this;
    }
    /*
    TODO: Kolla om det här verkligen är rätt sätt att overloada ==
    */
    bool operator==(const real_node_iterator &other)
    {
        return m_pointer == other.m_pointer;
    }

    bool operator!=(const real_node_iterator &other)
    {
        return m_pointer != other.m_pointer;
    }

    T &operator*() // Kanske ska den här operatorn vara const
    {
        if (m_pointer == nullptr)
            throw std::out_of_range("Out of range");
        return m_pointer->data;
    }

    int *operator->()
    {
        return m_pointer;
    }

    real_node_iterator &forward_iteration()
    {
        S set_key_val = m_pointer->key;
        if (m_pointer->right != nullptr)
        {
            Node<S, T> *current = m_pointer->right;
            while (current->left != nullptr)
                current = current->left;
            m_pointer = current;
            return *this;
        }
        Node<S, T> *current = m_pointer;
        Node<S, T> *parent = current->parent;

        while (parent != nullptr && (parent->key < set_key_val))
        {
            current = parent;
            parent = parent->parent;

            if (parent == nullptr)
                break;
        }
        m_pointer = parent;
        return *this;
    }

    real_node_iterator &backward_iteration()
    {
        S set_key_val = m_pointer->key;
        if (m_pointer->left != nullptr)
        {
            Node<S, T> *current = m_pointer->left;
            while (current->right != nullptr)
                current = current->right;
            m_pointer = current;
            return *this;
        }
        Node<S, T> *current = m_pointer;
        Node<S, T> *parent = current->parent;

        while (parent != nullptr && (parent->key > set_key_val))
        {
            current = parent;
            parent = parent->parent;

            if (parent == nullptr)
                break;
        }
        m_pointer = parent;
        return *this;
    }

    real_node_iterator &operator++() // ++it
    {
        if (reversed)
            return backward_iteration();
        else
            return forward_iteration();
    }

    real_node_iterator operator++(int) // it++
    {
        if (reversed)
        {
            real_node_iterator rni = backward_iteration();
            return rni;
            //return backward_iteration();
        }
        else
        {
            real_node_iterator rni = forward_iteration();
            return rni;
            //return forward_iteration();
        }
    }

    friend void swap(real_node_iterator<S, T> &first, real_node_iterator<S, T> &second)
    {
        std::swap(first.m_pointer, second.m_pointer);
    }
};

// Const iterator

template <class S, class T>
struct Node;

template <class S, class T>
class const_real_node_iterator
{
public:
    Node<S, T> *m_pointer;
    bool reversed;
    bool is_const;

    const_real_node_iterator(Node<S, T> *pointer)
    {
        m_pointer = pointer;
        reversed = false;
        is_const = false;
    }

    const_real_node_iterator(Node<S, T> *pointer, bool reversed, bool is_const)
    {
        m_pointer = pointer;
        this->reversed = reversed;
        this->is_const = is_const;
    }

    const_real_node_iterator(const const_real_node_iterator &other)
    {
        m_pointer = other.m_pointer;
        reversed = other.reversed;
        is_const = other.is_const;
    }

    const_real_node_iterator &operator=(const const_real_node_iterator &other)
    {
        m_pointer = other.m_pointer;
        return *this;
    }
    /*
    TODO: Kolla om det här verkligen är rätt sätt att overloada ==
    */
    bool operator==(const const_real_node_iterator &other)
    {
        return m_pointer == other.m_pointer;
    }

    bool operator!=(const const_real_node_iterator &other)
    {
        return m_pointer != other.m_pointer;
    }

    const T &operator*() // Kanske ska den här operatorn vara const
    {
        if (m_pointer == nullptr)
            throw std::out_of_range("Out of range");
        return m_pointer->data;
    }

    int *operator->()
    {
        return m_pointer;
    }

    const_real_node_iterator &forward_iteration()
    {
        S set_key_val = m_pointer->key;
        if (m_pointer->right != nullptr)
        {
            Node<S, T> *current = m_pointer->right;
            while (current->left != nullptr)
                current = current->left;
            m_pointer = current;
            return *this;
        }
        Node<S, T> *current = m_pointer;
        Node<S, T> *parent = current->parent;

        while (parent != nullptr && (parent->key < set_key_val))
        {
            current = parent;
            parent = parent->parent;

            if (parent == nullptr)
                break;
        }
        m_pointer = parent;
        return *this;
    }

    const_real_node_iterator &backward_iteration()
    {
        S set_key_val = m_pointer->key;
        if (m_pointer->left != nullptr)
        {
            Node<S, T> *current = m_pointer->left;
            while (current->right != nullptr)
                current = current->right;
            m_pointer = current;
            return *this;
        }
        Node<S, T> *current = m_pointer;
        Node<S, T> *parent = current->parent;

        while (parent != nullptr && (parent->key > set_key_val))
        {
            current = parent;
            parent = parent->parent;

            if (parent == nullptr)
                break;
        }
        m_pointer = parent;
        return *this;
    }

    const_real_node_iterator &operator++() // ++it
    {
        if (reversed)
            return backward_iteration();
        else
            return forward_iteration();
    }

    const_real_node_iterator operator++(int) // it++
    {
        if (reversed)
        {
            const_real_node_iterator rni = backward_iteration();
            return rni;
            //return backward_iteration();
        }
        else
        {
            const_real_node_iterator rni = forward_iteration();
            return rni;
            //return forward_iteration();
        }
    }

    friend void swap(const_real_node_iterator<S, T> &first, const_real_node_iterator<S, T> &second)
    {
        std::swap(first.m_pointer, second.m_pointer);
    }
};

template <class S, class T>
struct Node
{
public:
    typedef real_node_iterator<S, T> iterator;       // TODO
    typedef real_node_iterator<S, T> const_iterator; // TODO

    S key;
    T data;
private:
    Node<S, T> *parent; // points to parent Node
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
    
    Node(S key_data, T value_data, Node<S,T> * parent_Node)
    {
        key = key_data;
        data = value_data;
        parent = parent_Node;
        right = nullptr;
        left = nullptr;
    }

    iterator begin()
    {
        Node<int, int> *current = this;
        while (current->left != nullptr)
            current = current->left;
        return iterator(current);
    }
    iterator end()
    {
        Node<int, int> *current = this;
        while (current->right != nullptr)
            current = current->right;
        current->right = nullptr;
        return iterator(current->right);
    }

    iterator rbegin()
    {
        Node<int, int> *current = this;
        while (current->right != nullptr)
            current = current->right;
        return iterator(current, true);
    }
    iterator rend()
    {
        Node<int, int> *current = this;
        while (current != nullptr)
            current = current->left;
        return iterator(current, true);
    }

    // friend declarations
    friend class real_node_iterator<S, T>;
    friend class const_real_node_iterator<S, T>;
    friend void insert<S, T>(Node<S, T> * & p, S key, T to_be_inserted);
    friend void insert<S, T>(Node<S, T> * & p, Node<S, T> * & z);
    friend void print_inorder<S, T>(Node<S, T> *p);
    friend void print_three_times();
};
#endif 
