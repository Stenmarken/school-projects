

// #include "Node.h"
#include <iterator>
#include <algorithm>
#include <iostream>
#include <stack>

using namespace std;

template <class S, class T>
struct Node;

template <class S, class T>
class real_node_iterator
{
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = Node<S, T>;
    using pointer = Node<S, T> *;
    using reference = Node<S, T> &;

    Node<S, T> *m_pointer;
    bool reversed;
    bool is_const;

    // Stack members
    stack<Node<S, T> *> m_stack;
    Node<S, T> *stack_pointer;
    bool empty_once = false;
    Node<S, T> *root;

public:
    void clear_stack()
    {
        while (!m_stack.empty())
        {
            m_stack.pop();
        }
    }

    bool has_next_node()
    {
        // Om stacken är tom och empty_once inte är sann, returnera true
        if (m_stack.empty() && !empty_once)
        {
            empty_once = true;
            return true;
        }
        return !m_stack.empty();
    }

    real_node_iterator &next_node()
    {
        if (!m_stack.empty())
        {
            stack_pointer = m_stack.top();
            m_stack.pop();
            add_left_nodes_stack(stack_pointer->right);
        }
        return *this;
    }

    void add_left_nodes_stack(Node<S, T> *p)
    {
        while (p != nullptr)
        {
            m_stack.push(p);
            p = p->left;
        }
    }

    real_node_iterator(Node<S, T> *pointer, Node<S, T> *root)
    {
        m_pointer = pointer;
        stack_pointer = pointer;
        reversed = false;
        is_const = false;
        clear_stack();
        add_left_nodes_stack(root);
        next_node();
    }

    real_node_iterator(Node<S, T> *pointer, bool reversed, bool is_const, Node<S, T> *root)
    {
        m_pointer = pointer;
        stack_pointer = pointer;
        this->reversed = reversed;
        this->is_const = is_const;
        clear_stack();
        add_left_nodes_stack(root);
        next_node();
    }

    real_node_iterator(const real_node_iterator &other)
    {
        m_pointer = other.m_pointer;
        reversed = other.reversed;
        is_const = other.is_const;
        clear_stack();
        m_stack = other.m_stack;
    }

    real_node_iterator &operator=(const real_node_iterator &other)
    {
        m_pointer = other.m_pointer;
        reversed = other.reversed;
        is_const = other.is_const;
        clear_stack();
        m_stack = other.m_stack;
        return *this;
    }

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

    T &operator()()
    {
        if (stack_pointer == nullptr)
            throw std::out_of_range("Out of range");
        return stack_pointer->data;
    }

    int *operator->()
    {
        return m_pointer;
    }

    real_node_iterator &forward_iteration()
    {
        if (m_pointer->right != nullptr)
        {
            Node<S, T> *current = m_pointer->right;
            while (current->left != nullptr)
                current = current->left;
            m_pointer = current;
            return *this;
        }

        Node<S, T> *parent = m_pointer->parent;

        while (parent != nullptr && m_pointer == parent->right)
        {
            m_pointer = parent;
            parent = parent->parent;
        }
        m_pointer = parent;
        return *this;
    }

    real_node_iterator &backward_iteration()
    {
        if (m_pointer->left != nullptr)
        {
            Node<S, T> *current = m_pointer->left;
            while (current->right != nullptr)
                current = current->right;
            m_pointer = current;
            return *this;
        }

        Node<S, T> *parent = m_pointer->parent;

        while (parent != nullptr && m_pointer == parent->left)
        {
            m_pointer = parent;
            parent = parent->parent;
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

    real_node_iterator &operator++(int) // it++
    {
        if (reversed)
            return backward_iteration();
        else
            return forward_iteration();
    }

    S get_key()
    {
        return m_pointer->key;
    }

    T get_data()
    {
        return m_pointer->data;
    }

    Node<S, T> *get_pointer()
    {
        return m_pointer;
    }

    void set_key(S key)
    {
        m_pointer->key = key;
    }

    void set_data(T data)
    {
        m_pointer->data = data;
    }

    friend void swap(real_node_iterator<S, T> &first, real_node_iterator<S, T> &second)
    {
        std::swap(first.m_pointer, second.m_pointer);
    }
};
