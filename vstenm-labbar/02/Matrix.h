//
// DD1388 - Lab 2: The matrix
//
#ifndef MATRIX_H
#define MATRIX_H

#include <initializer_list>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <cstring>
#include <algorithm>

template <typename T>
class Matrix
{

    /* static_assert( ... fill in the condition ..., "...") // T must be move-constructible */
    /* static_assert( ... fill in the condition ..., "...") // T must be move-assignable */

public:
    // constructors and assignment operators
    /* TODO: Make the appropriate constructor(s) explicit */
    Matrix();
    Matrix(size_t dim);
    Matrix(size_t rows, size_t cols);
    Matrix(const std::initializer_list<T> &list);
    Matrix(Matrix<T> &other);
    Matrix(const Matrix<T> &other);
    Matrix(Matrix<T> &&other) noexcept;

    Matrix<T> &operator=(const Matrix<T> &other);
    Matrix<T> &operator=(Matrix<T> &&other) noexcept;

    ~Matrix();

    // setter
    void set(int index, T value);

    // getter
    T get(int index);

    // accessors
    size_t rows() const;
    size_t cols() const;

    T &operator()(size_t row, size_t col);
    const T &operator()(size_t row, size_t col) const;

    // operators
    Matrix<T> operator*(const Matrix<T> &other) const;
    Matrix<T> operator+(const Matrix<T> &other) const;
    Matrix<T> operator-(const Matrix<T> &other) const;

    void operator*=(const Matrix<T> &other);
    void operator+=(const Matrix<T> &other);
    void operator-=(const Matrix<T> &other);

    // methods
    void reset();

    void insert_row(size_t row);
    void append_row(size_t row);
    void remove_row(size_t row);
    void insert_column(size_t col);
    void append_column(size_t col);
    void remove_column(size_t col);

    void print();
    bool compare_matrices(T compare_vec[], int compare_vec_capacity);

    // iterators
    typedef T *iterator;

    iterator begin();
    iterator end();

    // size_t m_rows;
private:
    size_t m_rows;
    size_t m_cols;
    size_t m_capacity;
    T *m_vec;
};

// input/output operators
template <typename T>
std::istream &operator>>(std::istream &is, Matrix<T> &m);

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &m);

// functions
template <typename T>
Matrix<T> identity(size_t dim);

template <typename T>
Matrix<T>::Matrix()
{
    static_assert(std::is_move_constructible<T>::value, "Not move constructible");
    static_assert(std::is_move_assignable<T>::value, "Not move assignable");
    m_vec = {};
    m_rows = 0;
    m_cols = 0;
    m_capacity = 0;
}

template <typename T>
Matrix<T>::Matrix(size_t dim)
{
    static_assert(std::is_move_constructible<T>::value, "Not move constructible");
    static_assert(std::is_move_assignable<T>::value, "Not move assignable");
    m_vec = new T[dim * dim];
    for (int i = 0; i < dim * dim; i++)
        m_vec[i] = 0;
    m_rows = dim;
    m_cols = dim;
    m_capacity = dim * dim;
}

template <typename T>
void Matrix<T>::print()
{
    std::cout << "\n"
              << std::endl;
    for (int i = 0; i < m_capacity; i++)
    {
        std::cout << m_vec[i] << " ";
        if ((i + 1) % m_cols == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename T>
bool Matrix<T>::compare_matrices(T compare_vec[], int compare_vec_capacity)
{
    if (compare_vec_capacity != m_capacity)
        return false;
    for (int i = 0; i < m_capacity; i++)
    {
        if (m_vec[i] != compare_vec[i])
            return false;
    }
    return true;
}

template <typename T>
Matrix<T>::Matrix(size_t rows, size_t cols)
{
    static_assert(std::is_move_constructible<T>::value, "Not move constructible");
    static_assert(std::is_move_assignable<T>::value, "Not move assignable");
    m_vec = new T[rows * cols];
    for (int i = 0; i < rows * cols; i++)
        m_vec[i] = 0;
    m_rows = rows;
    m_cols = cols;
    m_capacity = rows * cols;
}

template <typename T>
Matrix<T>::Matrix(const std::initializer_list<T> &list)
{
    static_assert(std::is_move_constructible<T>::value, "Not move constructible");
    static_assert(std::is_move_assignable<T>::value, "Not move assignable");

    // The trick below is taken from
    // https://stackoverflow.com/questions/22239097/determining-if-square-root-is-an-integer
    int s = sqrt(list.size());
    if ((s * s) != list.size())
    {
        throw std::out_of_range("The size of the initializer_list must be an even square root");
    }

    m_vec = new T[list.size()];
    m_rows = s;
    m_cols = s;
    m_capacity = list.size();

    int value = *list.begin();
    int row = 0;
    int col = 0;
    int counter = 0;
    std::initializer_list<int>::iterator it;

    for (auto element : list)
    {
        m_vec[counter] = element;
        counter++;
    }
}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &other)
{
    static_assert(std::is_move_constructible<T>::value, "Not move constructible");
    static_assert(std::is_move_assignable<T>::value, "Not move assignable");

    m_vec = new T[other.m_capacity];
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_capacity = other.m_capacity;

    for (int i = 0; i < other.m_capacity; i++)
    {
        m_vec[i] = other.m_vec[i];
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &other)
{
    static_assert(std::is_move_constructible<T>::value, "Not move constructible");
    static_assert(std::is_move_assignable<T>::value, "Not move assignable");

    m_vec = new T[other.m_capacity];
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_capacity = other.m_capacity;

    for (int i = 0; i < other.m_capacity; i++)
    {
        m_vec[i] = other.m_vec[i];
    }
}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &&other) noexcept
{
    static_assert(std::is_move_constructible<T>::value, "Not move constructible");
    static_assert(std::is_move_assignable<T>::value, "Not move assignable");

    m_vec = other.m_vec;
    m_rows = other.m_rows;
    m_cols = other.m_cols;
    m_capacity = other.m_capacity;

    other.m_vec = nullptr;
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
{
    if (this != &other)
    {
        T *temp = new T[other.m_capacity];
        std::copy(other.m_vec, other.m_vec + other.m_capacity, temp);

        delete[] m_vec;

        m_vec = temp;
        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_capacity = other.m_capacity;
    }
    return *this;
}

template <typename T>
Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other) noexcept
{
    if (this != &other)
    {
        delete[] m_vec;
        m_vec = other.m_vec;

        m_rows = other.m_rows;
        m_cols = other.m_cols;
        m_capacity = other.m_capacity;

        other.m_cols = 0;
        other.m_rows = 0;
        other.m_capacity = 0;
        other.m_vec = nullptr;
    }
    return *this;
}

template <typename T>
Matrix<T>::~Matrix()
{
    delete[] m_vec;
}

template <typename T>
void Matrix<T>::set(int index, T value)
{
    m_vec[index] = value;
}

template <typename T>
T Matrix<T>::get(int index)
{
    return m_vec[index];
}

template <typename T>
size_t Matrix<T>::rows() const
{
    return m_rows;
}

template <typename T>
size_t Matrix<T>::cols() const
{
    return m_cols;
}

template <typename T>
T &Matrix<T>::operator()(size_t row, size_t col)
{
    return m_vec[row * m_cols + col];
}

template <typename T>
const T &Matrix<T>::operator()(size_t row, size_t col) const
{
    return m_vec[row * m_cols + col];
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &other) const
{
    if (m_cols != other.rows())
    {
        throw std::invalid_argument("The dimensions of the matrices don't match");
    }
    Matrix<T> m = Matrix<T>(m_rows, other.cols());

    // For the matrix multiplication part. I took inspiration from this article
    // https://www.programiz.com/cpp-programming/examples/matrix-multiplication
    for (int i = 0; i < m_rows; i++)
        for (int j = 0; j < other.cols(); j++)
            for (int k = 0; k < m_cols; k++)
                m(i, j) += operator()(i, k) * other(k, j);

    return m;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const
{
    if (m_rows != other.rows() || m_cols != other.cols())
    {
        throw std::invalid_argument("The dimensions of the matrices don't match");
    }
    Matrix<T> m = Matrix<T>(m_rows, m_cols);
    for (int i = 0; i < m_capacity; i++)
    {
        m.m_vec[i] = m_vec[i] + other.m_vec[i];
    }
    return m;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &other) const
{
    if (m_rows != other.rows() || m_cols != other.cols())
    {
        throw std::invalid_argument("The dimensions of the matrices don't match");
    }
    Matrix<T> m = Matrix<T>(m_rows, m_cols);
    for (int i = 0; i < m_capacity; i++)
    {
        m.m_vec[i] = m_vec[i] - other.m_vec[i];
    }
    return m;
}

template <typename T>
void Matrix<T>::operator*=(const Matrix<T> &other)
{
    if (m_cols != other.rows())
    {
        throw std::invalid_argument("The dimensions of the matrices don't match");
    }

    Matrix<T> m = Matrix<T>(m_rows, other.cols());
    // For the matrix multiplication part. I took inspiration from this article
    // https://www.programiz.com/cpp-programming/examples/matrix-multiplication
    for (int i = 0; i < m_rows; i++)
        for (int j = 0; j < other.cols(); j++)
            for (int k = 0; k < m_cols; k++)
                m(i, j) += operator()(i, k) * other(k, j);

    //*this = m;
}

template <typename T>
void Matrix<T>::operator+=(const Matrix<T> &other)
{
    if (m_rows != other.rows() || m_cols != other.cols())
    {
        throw std::invalid_argument("The dimensions of the matrices don't match");
    }
    for (int i = 0; i < m_capacity; i++)
    {
        m_vec[i] += other.m_vec[i];
    }
}

template <typename T>
void Matrix<T>::operator-=(const Matrix<T> &other)
{
    if (m_rows != other.rows() || m_cols != other.cols())
    {
        throw std::invalid_argument("The dimensions of the matrices don't match");
    }
    for (int i = 0; i < m_capacity; i++)
    {
        m_vec[i] -= other.m_vec[i];
    }
}

template <typename T>
void Matrix<T>::reset()
{
    for (int i = 0; i < m_capacity; i++)
        m_vec[i] = T();
}

template <typename T>
void Matrix<T>::insert_row(size_t row)
{
    int old_capacity = m_capacity;
    m_capacity += m_cols;
    T *new_m_vec = new T[m_capacity];

    memcpy(new_m_vec, m_vec, old_capacity * sizeof(T));
    delete[] m_vec;
    m_vec = new_m_vec;
    m_rows++;

    int start_index = (row - 1) * m_cols; // m_cols is the number of cols in a row

    for (int i = m_capacity - 1; i >= start_index; i--)
        m_vec[i] = m_vec[i - m_cols];

    for (int i = start_index; i < start_index + m_cols; i++)
        m_vec[i] = 0;
}

template <typename T>
void Matrix<T>::append_row(size_t row)
{
    int old_capacity = m_capacity;
    m_capacity += m_cols;
    T *new_m_vec = new T[m_capacity];

    memcpy(new_m_vec, m_vec, old_capacity * sizeof(T));
    delete[] m_vec;
    m_vec = new_m_vec;
    m_rows++;

    int start_index = (row)*m_cols; // m_cols is the number of cols in a row

    for (int i = m_capacity - 1; i >= start_index; i--)
        m_vec[i] = m_vec[i - m_cols];

    for (int i = start_index; i < start_index + m_cols; i++)
        m_vec[i] = 0;
}

template <typename T>
void Matrix<T>::remove_row(size_t row)
{
    T *new_m_vec = new T[m_capacity - m_cols];
    int counter = 0;
    m_rows--;
    for (int i = 0; i < m_rows; i++)
    {
        if (i == row - 1)
        {
            counter += m_cols;
        }        
        for (int j = 0; j < m_cols; j++)
        {
            new_m_vec[i * m_cols + j] = m_vec[i * m_cols + j + counter];
        }
    }
    delete[] m_vec;
    m_capacity -= m_cols;
    m_vec = new_m_vec;
}

template <typename T>
void Matrix<T>::insert_column(size_t col)
{
    int old_capacity = m_capacity;
    m_capacity += m_rows;
    T *new_m_vec = new T[m_capacity];

    memcpy(new_m_vec, m_vec, old_capacity * sizeof(T));
    m_cols++;

    int counter = 1;
    for (int i = 0; i < m_rows; i++)
    {
        for (int j = 1; j <= m_cols; j++)
        {
            if (j == col)
            {
                new_m_vec[i * m_cols + j - 1] = 0;
                counter++;
            }
            else
                new_m_vec[i * m_cols + j - 1] = m_vec[i * m_cols + j - counter];
        }
    }

    delete[] m_vec;
    m_vec = new_m_vec;
}

template <typename T>
void Matrix<T>::append_column(size_t col)
{
    insert_column(col + 1);
}

template <typename T>
void Matrix<T>::remove_column(size_t col)
{
    int old_capacity = m_capacity;
    m_capacity -= m_rows;
    T *new_m_vec = new T[m_capacity];
    int counter = 1;
    int separate_counter = 0;

    for (int i = 0; i < m_rows; i++)
    {
        for (int j = 1; j <= m_cols; j++)
        {
            if (j == col)
            {
                counter++;
                separate_counter++;
                continue;
            }
            new_m_vec[i * m_cols + j - counter] = m_vec[i * m_cols + j - counter + separate_counter];
        }
    }
    m_cols--;
    delete[] m_vec;
    m_vec = new_m_vec;
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::begin()
{
    return iterator(&m_vec[0]);
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::end()
{
    return iterator(&m_vec[m_capacity]);
}

template <typename T>
std::istream &operator>>(std::istream &is, Matrix<T> &m)
{
    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < m.cols(); j++)
            is >> m(i, j);
    }
    return is;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &m)
{
    std::cout << std::endl;
    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < m.cols(); j++)
            os << m(i, j) << " ";
        os << std::endl;
    }
    os << std::endl;
    return os;
}

template <typename T>
Matrix<T> identity(size_t dim)
{
    Matrix<T> m = Matrix<T>(dim);
    for (int i = 0; i < m.rows() * m.cols(); i++)
    {
        if (i % (dim + 1) == 0)
            m.set(i, 1);
    }
    return m;
}

#endif // MATRIX_H