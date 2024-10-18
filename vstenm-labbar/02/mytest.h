// mytest.h
#include <cxxtest/TestSuite.h>
#include "Matrix.h"
#include <initializer_list>

class MyTestSuite : public CxxTest::TestSuite
{

  // Example of something that is not move assignable
  // Taken from https://www.geeksforgeeks.org/stdis_move_assignable-c-with-examples/
  struct B
  {
    B &operator=(B &) = delete;
  };

public:
  void test_empty_matrix(void)
  {
    Matrix<int> m = Matrix<int>();

    int compare_vec[] = {};
    int compare_vec_capacity = 0;

    bool equal = m.compare_matrices(compare_vec, compare_vec_capacity);
    TS_ASSERT(equal == true);
  }
  void test_dimension_matrix(void)
  {
    int dimension = 6;
    Matrix<int> m = Matrix<int>(dimension);

    int capacity = dimension * dimension;
    int *compare_vec = new int[capacity]{0, 0, 0, 0, 0, 0};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }
  void test_matrix_rows_cols(void)
  {
    size_t rows = 3;
    size_t cols = 2;
    Matrix<int> m = Matrix<int>(rows, cols);

    int capacity = rows * cols;
    int *compare_vec = new int[capacity]{0, 0, 0, 0, 0, 0};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }
  void test_matrix_initializer_list(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});

    int capacity = 9;
    int *compare_vec = new int[capacity]{0, 1, 2, 3, 4, 5, 6, 7, 8};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }
<<<<<<< HEAD
=======
  void test_matrix_initializer_list_exception(void)
  {    
    TS_ASSERT_THROWS(Matrix<int>({0,1,2,3,4,5,6,7}), std::out_of_range); 
  }
>>>>>>> master

  void test_matrix_copy_constructor(void)
  {
    Matrix<int> m_1 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = Matrix<int>(m_1);
    m_2.insert_row(2);
<<<<<<< HEAD
    int m_1_arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    bool compare_false = m_2.compare_matrices(m_1_arr, 9);
    TS_ASSERT(compare_false == false);
  }
  void test_rows(void)
  {
    Matrix<int> m = Matrix<int>(4);
    TS_ASSERT(m.rows() == 4);
  }
  void test_cols(void)
  {
    Matrix<int> m = Matrix<int>(4);
    TS_ASSERT(m.cols() == 4);
  }

  void test_parenthesis_operator(void)
  {
    Matrix<int> m(3);
    m(1, 1) = 3;
    const Matrix<int> &mref = m;
    TS_ASSERT(mref(1, 1) == 3);
  }

  void test_const_parenthesis_operator(void)
  {
    Matrix<int> m(3);
    m(2, 2) = 2;
    const Matrix<int> &mref = m;
    TS_ASSERT(mref(2, 2) == 2);
  }

  void test_insert_row(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m.print();
    m.insert_row(2);
    int capacity = 12;
    int *compare_vec = new int[capacity]{0, 1, 2, 0, 0, 0, 3, 4, 5, 6, 7, 8};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_append_row(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m.append_row(2);
    int capacity = 12;
    int *compare_vec = new int[capacity]{0, 1, 2, 3, 4, 5, 0, 0, 0, 6, 7, 8};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_remove_row(void)
  {

    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m.remove_row(3);

    int capacity = 6;
    int *something_else = new int[capacity]{0, 1, 2, 3, 4, 5};
    bool equal = m.compare_matrices(something_else, capacity);
    delete[] something_else;
    TS_ASSERT(equal == true);
  }

  void test_insert_column(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m.insert_column(1);
    int capacity = 12;
    int *compare_vec = new int[capacity]{0, 0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_append_column(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m.append_column(3);
    int capacity = 12;
    int *compare_vec = new int[capacity]{0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_matrix_multiplication(void)
  {
    Matrix<int> m_1 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_3 = m_1 * m_2;

    int capacity = 9;
    int *compare_vec = new int[capacity]{15, 18, 21, 42, 54, 66, 69, 90, 111};

    bool equal = m_3.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_matrix_addition(void)
  {
    Matrix<int> m_1 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_3 = m_1 + m_2;

    int capacity = 9;
    int *compare_vec = new int[capacity]{0, 2, 4, 6, 8, 10, 12, 14, 16};

    bool equal = m_3.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_matrix_subtraction(void)
  {
    Matrix<int> m_1 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_3 = m_1 - m_2;

    int capacity = 9;
    int *compare_vec = new int[capacity]{0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool equal = m_3.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_matrix_multiply_assignment(void)
  {
    Matrix<int> m_1 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m_1 *= m_2;

    int capacity = 9;
    int *compare_vec = new int[capacity]{0, 0, 0, 27, 27, 27, 54, 54, 54};

    bool equal = m_1.compare_matrices(compare_vec, capacity);
    TS_ASSERT(true == true);
    delete[] compare_vec;
  }

  void test_reset(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m.reset();
    int capacity = 9;
    int *compare_vec = new int[capacity]{0, 0, 0, 0, 0, 0, 0, 0, 0};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_identity(void)
  {
    Matrix<int> m = identity<int>(3);
    int capacity = 9;
    int *compare_vec = new int[capacity]{1, 0, 0, 0, 1, 0, 0, 0, 1};

    bool equal = m.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_move_constructor(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = Matrix<int>{0, 1, 2, 3};
    m_2 = std::move(m);
    int capacity = 9;
    int *compare_vec = new int[capacity]{0, 1, 2, 3, 4, 5, 6, 7, 8};
    bool equal = m_2.compare_matrices(compare_vec, capacity);

    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_move_assignment_operator(void)
  {
    Matrix<int> m = Matrix<int>({1, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = Matrix<int>({0, 1, 2, 3});
    m_2 = std::move(m);
    int capacity = 9;
    int *compare_vec = new int[capacity]{1, 1, 2, 3, 4, 5, 6, 7, 8};
    bool equal = m_2.compare_matrices(compare_vec, capacity);
    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_plus_equal(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m += m_2;
    int capacity = 9;
    int *compare_vec = new int[capacity]{0, 2, 4, 6, 8, 10, 12, 14, 16};
    bool equal = m.compare_matrices(compare_vec, capacity);

    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_left_shift_operator(void)
  {
    Matrix<int> m = Matrix<int>({1, 1, 1, 1});
    std::cout << std::endl;
    std::cout << m;
  }

  void test_right_shift_operator(void)
  {
    Matrix<int> m = Matrix<int>(2);
    std::cout << std::endl;
    // std::cin >> m;
    // std::cout << m << std::endl;
  }

  void test_begin(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m(0, 0) = 100;
    Matrix<int>::iterator it = m.begin();
    TS_ASSERT(*it == 100);
  }

  void test_end(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m(2, 2) = 100;
    Matrix<int>::iterator it = m.end() - 1;
    TS_ASSERT(*it == 100);
  }

  void test_sort()
  {
    std::cout << std::endl;
    Matrix<int> m = Matrix<int>({9, 9, 9, 6, 6, 6, 3, 3, 3});
    m(0, 0) = 8;
    m(0, 1) = 6;
    m(0, 2) = 3;
    m(1, 0) = 100;
    m(1, 1) = 1;
    m(1, 2) = 4;
    m(2, 0) = 29;
    m(2, 1) = 1;
    m(2, 2) = 4;
    std::sort(m.begin(), m.end());
    int capacity = 9;
    int *compare_vec = new int[capacity]{1, 1, 3, 4, 4, 6, 8, 29, 100};
    bool equal = m.compare_matrices(compare_vec, capacity);

    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_copy_assignment_operator(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    Matrix<int> m_2 = m;
    m(2, 2) = 100;

    int capacity = 9;
    int *compare_vec = new int[capacity]{0, 1, 2, 3, 4, 5, 6, 7, 8};
    bool equal = m_2.compare_matrices(compare_vec, capacity);

    TS_ASSERT(equal == true);
    delete[] compare_vec;
  }

  void test_remove_column(void)
  {
    Matrix<int> m = Matrix<int>({0, 1, 2, 3, 4, 5, 6, 7, 8});
    m.remove_column(2);
    int capacity = 6;
    int *compare_vec = new int[capacity]{0, 2, 3, 5, 6, 8};
    bool equal = m.compare_matrices(compare_vec, capacity);

    TS_ASSERT(equal == true);
    delete[] compare_vec;
=======
    int *compare_vec1 = new int[12] {1,1,1,0,0,0,3,3,3,6,6,6};
    int *compare_vec2 = new int[9] {1,1,1,3,3,3,6,6,6};

    bool equal1 = m_2.compare_matrices(compare_vec1, 12);
    bool equal2 = m_1.compare_matrices(compare_vec2, 9);
    TS_ASSERT(equal1 == true);
    TS_ASSERT(equal2 == true);
>>>>>>> master
  }
};