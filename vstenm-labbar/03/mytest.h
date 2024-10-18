// mytest.h
#include <cxxtest/TestSuite.h>
#include "Complex.h"
#include <initializer_list>
#include <string>


using namespace std;
class MyTestSuite : public CxxTest::TestSuite
{

public:
  void test_null(void)
  {
    Complex x;
    TS_ASSERT(x.real() == 0 && x.imag() == 0);
  }

  void test_assignment_operator_real(void)
  {
    Complex x = 5;
    TS_ASSERT(x.real() == 5 && x.imag() == 0);
  }

  void test_assignment_operator_imag(void)
  {
    Complex x = 5_i;
    TS_ASSERT(x.real() == 0 && x.imag() == 5);
  }

  void test_assignment_operator_complex(void)
  {
    Complex x = 5 + 5_i;
    TS_ASSERT(x.real() == 5 && x.imag() == 5);
  }

  void test_real(void)
  {
    Complex x(10, -10);
    TS_ASSERT(x.real() == 10);
  }

  void test_imag(void)
  {
    Complex x(10, -10);
    TS_ASSERT(x.imag() == -10);
  }

  void test_constructor(void)
  {
    Complex y(6, 2);
    TS_ASSERT(y.real() == 6 && y.imag() == 2);
  }

  void test_copy_constructor(void)
  {
    Complex x(6, 2);
    Complex y(x);
    TS_ASSERT(y.real() == 6 && y.imag() == 2);
  }

  void test_abs_both_positive(void)
  {
    TS_ASSERT(abs(Complex(2, 2)) == 2*sqrt(2));
  }

  void test_abs_both_negative(void)
  {
    TS_ASSERT(abs(Complex(-2, -2)) == 2*sqrt(2));
  }

  void test_abs_real_negative(void)
  {
    TS_ASSERT(abs(Complex(-2, 2)) == 2*sqrt(2));
  }

  void test_abs_imag_negative(void)
  {
    TS_ASSERT(abs(Complex(2, -2)) == 2*sqrt(2));
  }

  void test_plus_equals_both_complex(void)
  {
    Complex x(6, 2);
    Complex y(3, 1);
    x += y;
    TS_ASSERT(x.real() == 9 && x.imag() == 3);
  }

  void test_plus_equals_real(void)
  {
    Complex x(6, 2);
    x += 3;
    TS_ASSERT(x.real() == 9 && x.imag() == 2);
  }

  void test_plus_equals_imag(void)
  {
    Complex x(6, 2);
    x += 3_i;
    TS_ASSERT(x.real() == 6 && x.imag() == 5);
  }

  void test_minus_equals_complex(void)
  {
    Complex x(6, 2);
    Complex y(3, 1);
    x -= y;
    TS_ASSERT(x.real() == 3 && x.imag() == 1);
  }

  void test_minus_equals_real(void)
  {
    Complex x(6, 2);
    x -= 3;
    TS_ASSERT(x.real() == 3 && x.imag() == 2);
  }

  void test_minus_equals_imag(void)
  {
    Complex x(6, 2);
    x -= 3_i;
    TS_ASSERT(x.real() == 6 && x.imag() == -1);
  }

  void test_times_equals(void)
  {
    Complex x(3, 2);
    Complex y(1, 4);
    x *= y;
    TS_ASSERT(x.real() == -5 && x.imag() == 14);
  }

  void test_divide_equals_easy(void)
  {
    Complex x(8, 4);
    Complex y(2, 1);
    x /= y;
    TS_ASSERT(x.real() == 4 && x.imag() == 0);
  }
  void test_divide_equals_harder(void)
  {
    Complex x(4, -2);
    Complex y(3, 5);
    x /= y;
    TS_ASSERT(x.real() == 1.0 / 17.0 && x.imag() == -13.0 / 17.0);
  }

  void test_unary_plus(void)
  {
    Complex x(6, 2);
    Complex y = +x;
    TS_ASSERT(y.real() == 6 && y.imag() == 2);
  }

  void test_unary_minus(void)
  {
    Complex x(6, 2);
    Complex y = -x;
    TS_ASSERT(y.real() == -6 && y.imag() == -2);
  }

  void test_plus_complex(void)
  {
    Complex x(6, 2);
    Complex y(3, 1);
    Complex z = x + y;
    TS_ASSERT(z.real() == 9 && z.imag() == 3);
  }

  void test_plus_complex_real_left(void)
  {
    Complex x(6, 2);
    Complex y = 3 + x;
    TS_ASSERT(y.real() == 9 && y.imag() == 2);
  }

  void test_plus_complex_real_right(void)
  {
    Complex x(6, 2);
    Complex y = x + 3;
    TS_ASSERT(y.real() == 9 && y.imag() == 2);
  }

  void test_minus_complex(void)
  {
    Complex x(6, 2);
    Complex y(3, 1);
    Complex z = x - y;
    TS_ASSERT(z.real() == 3 && z.imag() == 1);
  }

  void test_minus_complex_real_left(void)
  {
    Complex x(6, 2);
    Complex y = 3 - x;
    TS_ASSERT(y.real() == -3 && y.imag() == -2);
  }

  void test_minus_complex_real_right(void)
  {
    Complex x(6, 2);
    Complex y = x - 3;
    TS_ASSERT(y.real() == 3 && y.imag() == 2);
  }

  void test_multiply_complex(void)
  {
    Complex x(3, 2);
    Complex y(1, 4);
    Complex z = x * y;
    TS_ASSERT(z.real() == -5 && z.imag() == 14);
  }

  void test_multiply_complex_real_left(void)
  {
    Complex x(3, 2);
    Complex y = 2 * x;
    TS_ASSERT(y.real() == 6 && y.imag() == 4);
  }

  void test_multiply_complex_real_right(void)
  {
    Complex x(3, 2);
    Complex y = x * 2;
    TS_ASSERT(y.real() == 6 && y.imag() == 4);
  }

  void test_divide_complex(void)
  {
    Complex x(8, 4);
    Complex y(2, 1);
    Complex z = x / y;
    TS_ASSERT(z.real() == 4 && z.imag() == 0);
  }

  void test_divide_complex_real_left(void)
  {
    Complex x(8, 4);
    Complex y = 2 / x;
    TS_ASSERT(y.real() == 0.2 && y.imag() == -0.1);
  }

  void test_equality_complex(void)
  {
    Complex x(6, 2);
    Complex y(6, 2);
    TS_ASSERT(x == y);
  }

  void test_equality_real(void)
  {
    Complex x(6, 0);
    double y = 6;
    TS_ASSERT(x == y);
  }

  void test_equality_imag(void)
  {
    Complex x(0, 2);
    Complex y = 2_i;
    TS_ASSERT(x == y);
  }

  void test_inequality_complex(void)
  {
    Complex x(6, 2);
    Complex y(6, 3);
    TS_ASSERT(x != y);
  }

  void test_inequality_real(void)
  {
    Complex x(6, 0);
    double y = 7;
    TS_ASSERT(x != y);
  }

  void test_inequality_imag(void)
  {
    Complex x(0, 2);
    Complex y = 3_i;
    TS_ASSERT(x != y);
  }

  void test_i_operator_int(void)
  {
    Complex k = 3 + 5_i;
    TS_ASSERT(k.real() == 3 && k.imag() == 5);
    k -= 5 + 1_i * Complex(5, 3);
  }

  void test_i_operator_double(void)
  {
    Complex k = 0 + 2.5_i;
    TS_ASSERT(k.real() == 0 && k.imag() == 2.5);
  }

  void test_left_shift(void)
  {
    Complex x(6, 2);
    ostringstream out;
    out << x;
    TS_ASSERT(out.str() == "(6, 2i)");
  }

  void test_complicated_left_shift(void)
  {
    ostringstream out;
    out << Complex(6, 6) / 6;
    TS_ASSERT(out.str() == "(1, 1i)");
  }

  void test_right_shift_complex(void)
  {
    Complex x;
    istringstream in("(6,0)");
    in >> x;
    TS_ASSERT(x.real() == 6 && x.imag() == 0);
  }

  void test_right_shift_real_easy(void)
  {
    Complex x;
    istringstream in("4");
    in >> x;
    TS_ASSERT(x.real() == 4);
  }

  void test_right_shift_real_harder(void)
  {
    Complex x;
    istringstream in("(10)");
    in >> x;
    TS_ASSERT(x.real() == 10);
  }

  void test_chain_operations(void)
  {
    Complex x;
    Complex y(0, 2);
    x = y + 1; // Should work!
    TS_ASSERT(x.real() == 1 && x.imag() == 2);
    x = y + y + 1 + 5; // Should work!
    TS_ASSERT(x.real() == 6 && x.imag() == 4);
    x = 2 + y; // Should work!
    TS_ASSERT(x.real() == 2 && x.imag() == 2);
    x = y = 0; // Should work!
    TS_ASSERT(x.real() == 0 && x.imag() == 0);
    TS_ASSERT(y.real() == 0 && y.imag() == 0);
  }
};